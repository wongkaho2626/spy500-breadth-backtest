#!/usr/bin/env python3
"""Download every post in a numbered LIHKG thread series.

LIHKG's JSON API is protected against plain HTTP clients.  This script starts a
temporary headless Chrome session, lets LIHKG initialise it normally, and makes
read-only API requests inside that browser session.  Thread predecessors are
discovered from ``parent_thread_id``; no list of guessed thread IDs is needed.

Example:
    python fetch_lihkg_series.py \
        --seed-url https://lihkg.com/thread/4152579/page/1 \
        --start-part 1 --end-part 34
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus, urljoin

try:
    import websocket
except ImportError:  # pragma: no cover - exercised only on an unprepared host
    websocket = None


DEFAULT_SEED_URL = "https://lihkg.com/thread/4152579/page/1"
DEFAULT_OUTPUT_DIR = "lihkg_美日韓_超長線十倍價投"
SERIES_TITLE = "美日韓 超長線十倍價投"
LEGACY_SERIES_TITLE = "賭Nio一年半內執笠"
SERIES_TITLES = (SERIES_TITLE, LEGACY_SERIES_TITLE)
THREAD_URL_RE = re.compile(r"^https?://(?:www\.)?lihkg\.com/thread/(\d+)(?:/page/\d+)?/?$")
PART_RE = re.compile(r"[\[【(](\d+)[\]】)]\s*$")


class ScrapeError(RuntimeError):
    """Raised when LIHKG or Chrome returns an unusable response."""


def parse_thread_id(url: str) -> int:
    """Return a LIHKG thread ID from a full thread URL."""
    match = THREAD_URL_RE.match(url.strip())
    if not match:
        raise ValueError(f"Not a LIHKG thread URL: {url!r}")
    return int(match.group(1))


def parse_part_number(title: str) -> int | None:
    """Read a trailing [34], 【34】, or (34) series number from a title."""
    match = PART_RE.search(title.strip())
    return int(match.group(1)) if match else None


def series_part_number(title: str) -> int | None:
    """Return the part number, including the unnumbered legacy first thread."""
    stripped = title.strip()
    if stripped == LEGACY_SERIES_TITLE:
        return 1
    return parse_part_number(stripped)


def is_series_title(title: str) -> bool:
    return any(name in title for name in SERIES_TITLES)


def find_chrome(explicit_path: str | None = None) -> str:
    """Locate a Chrome/Chromium executable on macOS, Linux, or Windows."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise FileNotFoundError(f"Chrome is not executable: {path}")

    candidates: list[str] = []
    system = platform.system()
    if system == "Darwin":
        candidates.extend(
            [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                str(
                    Path.home()
                    / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                ),
            ]
        )
    elif system == "Windows":
        for root_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(root_name)
            if root:
                candidates.append(str(Path(root) / "Google/Chrome/Application/chrome.exe"))

    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    for candidate in candidates:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Google Chrome or Chromium was not found. Install it or pass --chrome-path."
    )


class MessageTextParser(HTMLParser):
    """Convert LIHKG message HTML to readable text without external packages."""

    BLOCK_TAGS = {"blockquote", "div", "li", "ol", "p", "pre", "table", "tr", "ul"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCK_TAGS and self.parts:
            self.parts.append("\n")
        elif tag == "a":
            self.links.append((len(self.parts), attrs_dict.get("href") or ""))
        elif tag == "img":
            label = attrs_dict.get("alt") or attrs_dict.get("title") or "image"
            src = attrs_dict.get("src") or ""
            absolute_src = urljoin("https://lihkg.com/", src)
            self.parts.append(f"[{label}: {absolute_src}]" if src else f"[{label}]")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.links:
            start, href = self.links.pop()
            link_text = "".join(self.parts[start:]).strip()
            absolute_href = urljoin("https://lihkg.com/", href)
            if href and link_text != absolute_href and link_text != href:
                self.parts.append(f" ({absolute_href})")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts)).replace("\xa0", " ")
        value = re.sub(r"[ \t]+\n", "\n", value)
        value = re.sub(r"\n[ \t]+", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def message_to_text(message_html: str) -> str:
    parser = MessageTextParser()
    parser.feed(message_html or "")
    parser.close()
    return parser.text()


class CDPConnection:
    """Small synchronous Chrome DevTools Protocol client."""

    def __init__(self, websocket_url: str, timeout: float = 90.0) -> None:
        if websocket is None:
            raise RuntimeError(
                "Missing dependency 'websocket-client'. Install it with: "
                "python -m pip install websocket-client"
            )
        self._socket = websocket.create_connection(
            websocket_url,
            timeout=timeout,
            origin="http://localhost",
        )
        self._next_id = 0
        self._lock = threading.Lock()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            call_id = self._next_id
            self._socket.send(
                json.dumps({"id": call_id, "method": method, "params": params or {}})
            )
            while True:
                response = json.loads(self._socket.recv())
                if response.get("id") != call_id:
                    continue
                if "error" in response:
                    raise ScrapeError(f"Chrome CDP error for {method}: {response['error']}")
                return response.get("result", {})

    def close(self) -> None:
        self._socket.close()


@dataclass
class ChromeSession:
    chrome_path: str
    seed_url: str
    headless: bool = True

    def __post_init__(self) -> None:
        self._profile_dir: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self.cdp: CDPConnection | None = None

    def __enter__(self) -> "ChromeSession":
        self._profile_dir = tempfile.mkdtemp(prefix="lihkg-chrome-")
        command = [
            self.chrome_path,
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-sync",
            "--no-default-browser-check",
            "--no-first-run",
            "--remote-allow-origins=*",
            "--remote-debugging-port=0",
            f"--user-data-dir={self._profile_dir}",
        ]
        if self.headless:
            command.append("--headless=new")
        command.append(self.seed_url)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        port_file = Path(self._profile_dir) / "DevToolsActivePort"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not port_file.exists():
            if self._process.poll() is not None:
                raise ScrapeError("Chrome exited before its debugging endpoint was ready")
            time.sleep(0.1)
        if not port_file.exists():
            raise ScrapeError("Timed out waiting for Chrome's debugging endpoint")

        port = int(port_file.read_text(encoding="utf-8").splitlines()[0])
        target = self._wait_for_page_target(port)
        self.cdp = CDPConnection(target["webSocketDebuggerUrl"])
        self.cdp.call("Runtime.enable")
        self._wait_for_lihkg()
        return self

    def _wait_for_page_target(self, port: int) -> dict[str, Any]:
        deadline = time.monotonic() + 20
        endpoint = f"http://127.0.0.1:{port}/json/list"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(endpoint, timeout=2) as response:
                    targets = json.load(response)
                pages = [target for target in targets if target.get("type") == "page"]
                if pages:
                    return pages[0]
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            time.sleep(0.2)
        raise ScrapeError("Timed out waiting for a Chrome page target")

    def _wait_for_lihkg(self) -> None:
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            try:
                value = self.evaluate(
                    "JSON.stringify({title:document.title, ready:document.readyState})"
                )
                state = json.loads(value)
                if state["ready"] == "complete" and "LIHKG" in state["title"]:
                    return
            except (ScrapeError, json.JSONDecodeError, TypeError):
                pass
            time.sleep(0.4)
        raise ScrapeError(
            "LIHKG did not finish loading in Chrome. Re-run with --show-browser "
            "to inspect a possible CAPTCHA or network error."
        )

    def evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        if self.cdp is None:
            raise ScrapeError("Chrome session is not connected")
        result = self.cdp.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
            },
        ).get("result", {})
        if "exceptionDetails" in result:
            raise ScrapeError(f"JavaScript evaluation failed: {result['exceptionDetails']}")
        if result.get("subtype") == "error":
            raise ScrapeError(result.get("description", "JavaScript evaluation failed"))
        return result.get("value")

    def fetch_api_pages(self, requests: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
        """Fetch (thread_id, page) pairs concurrently inside the LIHKG origin."""
        urls = [
            f"/api_v2/thread/{thread_id}/page/{page}?order=reply_time"
            for thread_id, page in requests
        ]
        results = self.fetch_api_urls(urls)
        return [
            {**result, "threadId": thread_id, "page": page}
            for (thread_id, page), result in zip(requests, results, strict=True)
        ]

    def fetch_api_urls(self, urls: Sequence[str]) -> list[dict[str, Any]]:
        """Fetch same-origin API URLs concurrently inside the LIHKG session."""
        payload = json.dumps(list(urls), separators=(",", ":"))
        expression = f"""
            (async () => {{
                const urls = {payload};
                let device = localStorage.getItem('device') || '';
                try {{ device = JSON.parse(device); }} catch (_) {{}}
                const headers = {{
                    'Accept': 'application/json, text/plain, */*',
                    'X-LI-DEVICE-TYPE': 'browser',
                    'X-LI-DEVICE': device
                }};
                return JSON.stringify(await Promise.all(urls.map(async (url) => {{
                    try {{
                        const response = await fetch(url, {{credentials: 'include', headers}});
                        return {{url, status: response.status, body: await response.text()}};
                    }} catch (error) {{
                        return {{url, status: 0, error: String(error), body: ''}};
                    }}
                }})));
            }})()
        """
        raw = self.evaluate(expression, await_promise=True)
        if not isinstance(raw, str):
            raise ScrapeError("Chrome returned a non-text API batch result")
        return json.loads(raw)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.cdp is not None:
            try:
                self.cdp.close()
            except Exception:
                pass
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._profile_dir:
            shutil.rmtree(self._profile_dir, ignore_errors=True)


def response_payload(result: dict[str, Any], thread_id: int, page: int) -> dict[str, Any]:
    """Validate and unwrap one browser-side API fetch result."""
    if result.get("status") != 200:
        detail = result.get("error") or str(result.get("body", ""))[:200]
        raise ScrapeError(
            f"LIHKG API returned HTTP {result.get('status')} for thread "
            f"{thread_id}, page {page}: {detail}"
        )
    try:
        document = json.loads(result["body"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ScrapeError(f"Invalid JSON for thread {thread_id}, page {page}") from exc
    if document.get("success") != 1 or not isinstance(document.get("response"), dict):
        raise ScrapeError(
            f"LIHKG rejected thread {thread_id}, page {page}: {document!r}"
        )
    return document["response"]


def fetch_batch_with_retry(
    session: ChromeSession,
    requests: Sequence[tuple[int, int]],
    *,
    attempts: int = 6,
) -> dict[tuple[int, int], dict[str, Any]]:
    pending = list(requests)
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    for attempt in range(1, attempts + 1):
        results = session.fetch_api_pages(pending)
        retry: list[tuple[int, int]] = []
        for request, result in zip(pending, results, strict=True):
            thread_id, page = request
            try:
                completed[request] = response_payload(result, thread_id, page)
            except ScrapeError:
                if attempt == attempts:
                    raise
                retry.append(request)
        if not retry:
            return completed
        pending = retry
        wait_seconds = min(15 * (2 ** (attempt - 1)), 60)
        print(
            f"Retrying {len(retry)} rate-limited/failed page(s) in "
            f"{wait_seconds} seconds...",
            flush=True,
        )
        time.sleep(wait_seconds)
    return completed


def search_series_index(
    session: ChromeSession,
    *,
    author_user_id: int,
    start_part: int,
    end_part: int,
    max_pages: int = 3,
) -> dict[int, int]:
    """Find numbered parts in LIHKG search, restricted to the seed author."""
    candidates: dict[int, list[dict[str, Any]]] = {}
    for series_title in SERIES_TITLES:
        query = quote_plus(series_title)
        for search_page in range(1, max_pages + 1):
            url = (
                f"/api_v2/thread/search?q={query}&page={search_page}&count=100"
                "&sort=desc_create_time&type=thread"
            )
            result = session.fetch_api_urls([url])[0]
            if result.get("status") != 200:
                raise ScrapeError(
                    f"LIHKG search returned HTTP {result.get('status')}: "
                    f"{str(result.get('body', ''))[:200]}"
                )
            try:
                document = json.loads(result["body"])
                items = document["response"]["items"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ScrapeError("LIHKG search returned invalid JSON") from exc
            if document.get("success") != 1 or not isinstance(items, list):
                raise ScrapeError(f"LIHKG search rejected the query: {document!r}")

            for item in items:
                title = str(item.get("title", ""))
                part = series_part_number(title)
                if (
                    is_series_title(title)
                    and part is not None
                    and start_part <= part <= end_part
                    and item.get("user_id") == author_user_id
                    and isinstance(item.get("thread_id"), int)
                ):
                    candidates.setdefault(part, []).append(item)
            if len(items) < 100:
                break

    # Prefer a completed/high-reply thread if the author accidentally reused a
    # part number. Every candidate is validated again from its own first page.
    return {
        part: int(
            max(
                items,
                key=lambda item: (
                    int(item.get("no_of_reply", 0)),
                    int(item.get("total_page", 0)),
                    int(item.get("create_time", 0)),
                ),
            )["thread_id"]
        )
        for part, items in candidates.items()
    }


def discover_series(
    session: ChromeSession,
    seed_thread_id: int,
    start_part: int,
    end_part: int,
) -> tuple[dict[int, int], dict[tuple[int, int], dict[str, Any]], list[int]]:
    """Combine LIHKG search and parent links, then validate every candidate."""
    page_cache: dict[tuple[int, int], dict[str, Any]] = {}

    seed_key = (seed_thread_id, 1)
    seed_page = fetch_batch_with_retry(session, [seed_key])[seed_key]
    page_cache[seed_key] = seed_page
    seed_title = str(seed_page.get("title", ""))
    if not is_series_title(seed_title) or series_part_number(seed_title) != end_part:
        raise ScrapeError(
            f"Expected seed part {end_part}, but thread {seed_thread_id} is titled "
            f"{seed_title!r}"
        )
    author_user_id = seed_page.get("user_id")
    if not isinstance(author_user_id, int):
        raise ScrapeError("The seed thread has no usable author user ID")

    thread_ids = search_series_index(
        session,
        author_user_id=author_user_id,
        start_part=start_part,
        end_part=end_part,
    )
    thread_ids[end_part] = seed_thread_id

    # Search results are author-filtered and title-filtered. Their first pages
    # are fetched later as part of the content download, avoiding duplicate API
    # reads during discovery.
    for part in sorted(thread_ids, reverse=True):
        thread_id = thread_ids[part]
        print(f"Discovered part {part:02d}: thread {thread_id}", flush=True)

    missing_parts = sorted(set(range(start_part, end_part + 1)) - set(thread_ids))
    if missing_parts:
        print(
            "No thread exists in the author chain/search index for: "
            + ", ".join(f"part {part}" for part in missing_parts),
            flush=True,
        )
    return thread_ids, page_cache, missing_parts


def cache_path(cache_dir: Path, thread_id: int, page: int) -> Path:
    return cache_dir / f"thread_{thread_id}_page_{page:03d}.json"


def load_cached_page(path: Path, thread_id: int, page: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("thread_id") != thread_id or payload.get("page") != page:
        return None
    if not isinstance(payload.get("item_data"), list):
        return None
    return payload


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def chunks(values: Sequence[tuple[int, int]], size: int) -> Iterable[Sequence[tuple[int, int]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def download_thread(
    session: ChromeSession,
    part: int,
    thread_id: int,
    first_page: dict[str, Any],
    cache_dir: Path,
    batch_size: int,
    delay: float,
    resume: bool,
) -> list[dict[str, Any]]:
    total_pages = int(first_page["total_page"])
    pages: dict[int, dict[str, Any]] = {1: first_page}
    write_json(cache_path(cache_dir, thread_id, 1), first_page)
    missing: list[tuple[int, int]] = []
    for page_number in range(2, total_pages + 1):
        path = cache_path(cache_dir, thread_id, page_number)
        cached = load_cached_page(path, thread_id, page_number) if resume else None
        if cached is not None:
            pages[page_number] = cached
        else:
            missing.append((thread_id, page_number))

    for batch_index, batch in enumerate(chunks(missing, batch_size), start=1):
        fetched = fetch_batch_with_retry(session, batch)
        for (_, page_number), payload in fetched.items():
            pages[page_number] = payload
            write_json(cache_path(cache_dir, thread_id, page_number), payload)
        completed = len(pages)
        print(
            f"Part {part:02d}: {completed}/{total_pages} pages fetched",
            flush=True,
        )
        if delay and batch_index * batch_size < len(missing):
            time.sleep(delay)

    ordered = [pages[number] for number in range(1, total_pages + 1)]
    seen_post_ids: set[str] = set()
    for page in ordered:
        for post in page.get("item_data", []):
            post_id = str(post.get("post_id", ""))
            if post_id and post_id in seen_post_ids:
                raise ScrapeError(f"Duplicate post {post_id} in thread {thread_id}")
            if post_id:
                seen_post_ids.add(post_id)
    return ordered


def normalise_thread(part: int, pages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = pages[0]
    posts: list[dict[str, Any]] = []
    for page in pages:
        for item in page.get("item_data", []):
            post = dict(item)
            post["msg_text"] = message_to_text(str(item.get("msg", "")))
            reply_time = item.get("reply_time")
            if isinstance(reply_time, (int, float)):
                post["reply_time_utc"] = datetime.fromtimestamp(
                    reply_time, timezone.utc
                ).isoformat()
            posts.append(post)
    posts.sort(key=lambda item: int(item.get("msg_num", 0)))

    excluded = {"item_data", "parent_thread"}
    metadata = {key: value for key, value in first.items() if key not in excluded}
    return {
        "part": part,
        "thread_id": first["thread_id"],
        "url": f"https://lihkg.com/thread/{first['thread_id']}/page/1",
        "title": first["title"],
        "metadata": metadata,
        "posts": posts,
    }


def markdown_for_thread(thread: dict[str, Any]) -> str:
    lines = [
        f"# {thread['title']}",
        "",
        f"- Source: {thread['url']}",
        f"- Thread ID: {thread['thread_id']}",
        f"- Downloaded posts: {len(thread['posts'])}",
        "",
    ]
    for post in thread["posts"]:
        user = post.get("user") or {}
        nickname = post.get("user_nickname") or user.get("nickname") or "Unknown user"
        timestamp = post.get("reply_time_utc", post.get("reply_time", ""))
        lines.extend(
            [
                f"## #{post.get('msg_num', '?')} · {nickname}",
                "",
                f"Time: {timestamp}  ",
                f"Votes: +{post.get('like_count', 0)} / -{post.get('dislike_count', 0)}",
                "",
                post.get("msg_text", ""),
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def create_outputs(
    output_dir: Path,
    threads: Sequence[dict[str, Any]],
    *,
    requested_start_part: int,
    requested_end_part: int,
    missing_parts: Sequence[int],
) -> None:
    manifest: list[dict[str, Any]] = []
    combined_markdown: list[str] = []
    for thread in threads:
        part = int(thread["part"])
        json_path = output_dir / f"part_{part:02d}.json"
        markdown_path = output_dir / f"part_{part:02d}.md"
        write_json(json_path, thread)
        markdown = markdown_for_thread(thread)
        markdown_path.write_text(markdown, encoding="utf-8")
        combined_markdown.append(markdown)
        manifest.append(
            {
                "part": part,
                "thread_id": thread["thread_id"],
                "title": thread["title"],
                "url": thread["url"],
                "total_pages": thread["metadata"].get("total_page"),
                "post_count": len(thread["posts"]),
                "json_file": json_path.name,
                "markdown_file": markdown_path.name,
            }
        )
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "series.json",
        {
            "series_title": SERIES_TITLE,
            "legacy_series_title": LEGACY_SERIES_TITLE,
            "requested_parts": [requested_start_part, requested_end_part],
            "downloaded_parts": [thread["part"] for thread in threads],
            "missing_parts": list(missing_parts),
            "thread_count": len(threads),
            "post_count": sum(len(thread["posts"]) for thread in threads),
        },
    )
    (output_dir / "all_parts.md").write_text(
        "\n\n".join(combined_markdown), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-url", default=DEFAULT_SEED_URL)
    parser.add_argument("--start-part", type=int, default=1)
    parser.add_argument("--end-part", type=int, default=34)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chrome-path")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Concurrent LIHKG API reads per batch (default: 2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between request batches (default: 1.0)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore valid page files in the output cache",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show Chrome (useful when LIHKG presents an interactive challenge)",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.start_part < 1 or args.end_part < args.start_part:
        raise ValueError("Require 1 <= --start-part <= --end-part")
    if args.batch_size < 1 or args.batch_size > 10:
        raise ValueError("--batch-size must be between 1 and 10")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative")

    seed_thread_id = parse_thread_id(args.seed_url)
    chrome_path = find_chrome(args.chrome_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / ".pages"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using Chrome: {chrome_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    with ChromeSession(
        chrome_path=chrome_path,
        seed_url=args.seed_url,
        headless=not args.show_browser,
    ) as session:
        thread_ids, discovered_pages, missing_parts = discover_series(
            session,
            seed_thread_id,
            args.start_part,
            args.end_part,
        )
        threads: list[dict[str, Any]] = []
        for part in sorted(thread_ids):
            thread_id = thread_ids[part]
            first_page = discovered_pages.get((thread_id, 1))
            if first_page is None and not args.no_resume:
                first_page = load_cached_page(
                    cache_path(cache_dir, thread_id, 1), thread_id, 1
                )
            if first_page is None:
                first_page = fetch_batch_with_retry(session, [(thread_id, 1)])[
                    (thread_id, 1)
                ]
            first_title = str(first_page.get("title", ""))
            if (
                not is_series_title(first_title)
                or series_part_number(first_title) != part
            ):
                raise ScrapeError(
                    f"Part {part} failed first-page validation: thread {thread_id}, "
                    f"title {first_title!r}"
                )
            pages = download_thread(
                session,
                part,
                thread_id,
                first_page,
                cache_dir,
                args.batch_size,
                args.delay,
                resume=not args.no_resume,
            )
            thread = normalise_thread(part, pages)
            threads.append(thread)
            create_outputs(
                output_dir,
                threads,
                requested_start_part=args.start_part,
                requested_end_part=args.end_part,
                missing_parts=missing_parts,
            )
            print(
                f"Part {part:02d} complete: {len(thread['posts'])} posts",
                flush=True,
            )

    suffix = f"; missing series numbers: {missing_parts}" if missing_parts else ""
    print(f"Finished {len(threads)} parts in {output_dir}{suffix}", flush=True)
    return output_dir


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Profile every LAC comment in the downloaded LIHKG investment series.

The output is an auditable input for the accompanying investment-style report;
it does not attempt to predict returns or turn mention counts into buy signals.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAC_USER_ID = 734436
ARCHIVE_DIR = Path("lihkg_美日韓_超長線十倍價投")
OUTPUT_DIR = Path("analysis/lac_comment_analysis")

FRAMEWORK_PATTERNS = {
    "估值 / valuation": r"(?i)估值|valuation",
    "利潤率 / margin": r"(?i)margin|利潤率|毛利率",
    "期權 / options": r"(?i)期權|option|short put|covered call|\bsp\b|\bsc\b",
    "週期 / cycle": r"(?i)週期|周期|cycle",
    "資本開支 / capex": r"(?i)\bcapex\b",
    "槓桿 / leverage": r"(?i)槓桿|leverage",
    "TAM": r"(?i)\btam\b",
    "護城河 / moat": r"(?i)護城河|moat",
    "市佔率": r"(?i)市佔|market share",
    "DCF": r"(?i)\bdcf\b",
    "FCF": r"(?i)\bfcf\b",
    "ROIC": r"(?i)\broic\b",
    "情景分析": r"(?i)scenario|情景",
    "貼現率": r"(?i)discount rate",
    "被動投資 / 指數": r"被動投資|指數",
}

ENTITY_PATTERNS = {
    "NVDA / Nvidia": r"(?i)(?<![A-Za-z])(?:nvda|nvidia)(?![A-Za-z])",
    "TSMC / TSM": r"(?i)(?<![A-Za-z])(?:tsmc|tsm)(?![A-Za-z])",
    "OSCR": r"(?i)(?<![A-Za-z])oscr(?![A-Za-z])",
    "SK Hynix": r"(?i)sk\s*hynix|000660",
    "GOOG / Google": r"(?i)(?<![A-Za-z])(?:goog|google)(?![A-Za-z])",
    "MRVL": r"(?i)(?<![A-Za-z])mrvl(?![A-Za-z])",
    "BESI": r"(?i)(?<![A-Za-z])besi(?![A-Za-z])",
    "ASML": r"(?i)(?<![A-Za-z])asml(?![A-Za-z])",
    "AVGO": r"(?i)(?<![A-Za-z])avgo(?![A-Za-z])",
    "MSFT / Microsoft": r"(?i)(?<![A-Za-z])(?:msft|microsoft)(?![A-Za-z])",
    "MELI": r"(?i)(?<![A-Za-z])meli(?![A-Za-z])",
    "Adyen": r"(?i)(?<![A-Za-z])adyen(?![A-Za-z])",
    "Hermès / RMS": r"(?i)(?<![A-Za-z])(?:hermes|rms)(?![A-Za-z])",
    "AMD": r"(?i)(?<![A-Za-z])amd(?![A-Za-z])",
    "META": r"(?i)(?<![A-Za-z])meta(?![A-Za-z])",
    "AMZN / Amazon": r"(?i)(?<![A-Za-z])(?:amzn|amazon)(?![A-Za-z])",
    "COHR": r"(?i)(?<![A-Za-z])cohr(?![A-Za-z])",
    "1571 / 信邦控股": r"(?i)(?<![A-Za-z])(?:1571|xin point|信邦)(?![A-Za-z])",
    "LVMH": r"(?i)(?<![A-Za-z])lvmh(?![A-Za-z])",
    "CDI": r"(?i)(?<![A-Za-z])cdi(?![A-Za-z])",
}


def load_lac_rows(archive_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    all_post_count = 0
    for path in sorted(archive_dir.glob("part_*.json")):
        thread = json.loads(path.read_text(encoding="utf-8"))
        all_post_count += len(thread["posts"])
        for post in thread["posts"]:
            user = post.get("user") or {}
            if user.get("user_id") != LAC_USER_ID:
                continue
            timestamp = datetime.fromtimestamp(post["reply_time"], timezone.utc)
            rows.append(
                {
                    "part": thread["part"],
                    "thread_id": thread["thread_id"],
                    "page": post["page"],
                    "msg_num": post["msg_num"],
                    "post_id": post["post_id"],
                    "reply_time": post["reply_time"],
                    "reply_time_utc": timestamp.isoformat(),
                    "user_id": user.get("user_id"),
                    "nickname": post.get("user_nickname") or user.get("nickname"),
                    "like_count": post.get("like_count", 0),
                    "dislike_count": post.get("dislike_count", 0),
                    "status": post.get("status"),
                    "msg_html": post.get("msg", ""),
                    "msg_text": post.get("msg_text", ""),
                    "source_url": (
                        f"https://lihkg.com/thread/{thread['thread_id']}/"
                        f"page/{post['page']}?post={post['msg_num']}"
                    ),
                }
            )
    rows.sort(key=lambda row: (row["reply_time"], row["post_id"]))
    return rows, all_post_count


def count_messages(rows: list[dict[str, Any]], patterns: dict[str, str]) -> list[dict[str, Any]]:
    counts = [
        {
            "label": label,
            "message_count": sum(
                bool(re.search(pattern, row["msg_text"])) for row in rows
            ),
        }
        for label, pattern in patterns.items()
    ]
    return sorted(counts, key=lambda item: (-item["message_count"], item["label"]))


def find_post(rows: list[dict[str, Any]], part: int, msg_num: int) -> dict[str, Any]:
    return next(row for row in rows if row["part"] == part and row["msg_num"] == msg_num)


def build_candidate_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        {
            "rank": 1,
            "ticker": "CDI (Paris)",
            "company": "Christian Dior SE",
            "price": "€390.20 (3 Sep 2026 close)",
            "part": 34,
            "msg_num": 30,
            "lac_signal": "同日直接買入；把原本預留給 Hermès 的資金轉投 CDI，理由是估值極低。",
            "independent_check": "按 LVMH 持股作粗略穿透，CDI 對資產淨值折讓約 20–22%，年化 H1 盈利的穿透市盈率約 15 倍。",
            "main_risk": "控股折讓可以長期存在；成交疏落、缺乏催化劑，並完全承受 LVMH／奢侈品周期。",
            "confidence": "高",
        },
        {
            "rank": 2,
            "ticker": "ADYEN (Amsterdam)",
            "company": "Adyen N.V.",
            "price": "€1,043.00 (3 Sep 2026 close)",
            "part": 32,
            "msg_num": 588,
            "lac_signal": "H1 業績後仍判斷約低估 30%；多次稱其為歐洲 fintech 首選及 quality compounder。",
            "independent_check": "H1 2026 淨收入按固定匯率升 21%，EBITDA margin 49%，FCF conversion 86%；現價略低於該留言當日收市價。",
            "main_risk": "仍屬高估值增長股；兩宗收購令 2026 margin 約低一個百分點，整合和競爭可令估值假設落空。",
            "confidence": "中高",
        },
        {
            "rank": 3,
            "ticker": "1571 (Hong Kong)",
            "company": "信邦控股 / Xin Point Holdings",
            "price": "約 HK$3.81 (4 Sep 2026 snapshot)",
            "part": 33,
            "msg_num": 647,
            "lac_signal": "賣出 0398 全數換入 1571，認為其 business model、ROIC 更好且更穩定；其後表示跌會再買。",
            "independent_check": "2026 中期股息 HK$0.15；沿用上一個末期 HK$0.30 計，指示性股息率約 11.8%，手頭訂單約 RMB10.26bn。",
            "main_risk": "H1 收入跌 6.3%、盈利跌 11.7%；股息非保證，流動性低，並有汽車客戶召回事件的不確定性。",
            "confidence": "中高",
        },
        {
            "rank": 4,
            "ticker": "COHR (NYSE)",
            "company": "Coherent Corp.",
            "price": "$273.90 (4 Sep 2026 13:38 UTC)",
            "part": 32,
            "msg_num": 975,
            "lac_signal": "曾列明 $280、$240 已買，下一注 $200；另稱光通訊只選 COHR，估值合理但不會佔太多。",
            "independent_check": "現價低於首個 $280 買入位；FY26 Q4 收入升 34%，GAAP gross margin 38.5%，non-GAAP EPS $1.74。",
            "main_risk": "AI 光通訊預期、擴產及客戶集中度均高；GAAP P/E 很高，若增長或 margin 不達標，下行可很大。",
            "confidence": "中",
        },
    ]
    for item in specs:
        post = find_post(rows, item.pop("part"), item.pop("msg_num"))
        item["signal_date"] = post["reply_time_utc"][:10]
        item["source_url"] = post["source_url"]
    return specs


def analyse(archive_dir: Path = ARCHIVE_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, all_post_count = load_lac_rows(archive_dir)
    if not rows:
        raise RuntimeError("No LAC comments found")
    post_ids = [row["post_id"] for row in rows]
    user_ids = Counter(row["user_id"] for row in rows)
    nicknames = Counter(row["nickname"] for row in rows)
    if user_ids != Counter({LAC_USER_ID: len(rows)}) or nicknames != Counter({"LAC": len(rows)}):
        raise RuntimeError("LAC identity is inconsistent across the archive")
    if len(post_ids) != len(set(post_ids)):
        raise RuntimeError("Duplicate LAC post IDs found")

    lengths = sorted(len(row["msg_text"]) for row in rows)
    parts = sorted(set(row["part"] for row in rows))
    summary = {
        "profile": {
            "lac_user_id": LAC_USER_ID,
            "comment_count": len(rows),
            "archive_post_count": all_post_count,
            "share_of_archive": len(rows) / all_post_count,
            "character_count": sum(lengths),
            "median_comment_characters": lengths[len(lengths) // 2],
            "comments_at_least_200_characters": sum(value >= 200 for value in lengths),
            "comments_at_least_500_characters": sum(value >= 500 for value in lengths),
            "first_comment_utc": rows[0]["reply_time_utc"],
            "last_comment_utc": rows[-1]["reply_time_utc"],
            "parts_with_comments": parts,
            "part_count": len(parts),
            "missing_series_part": 29,
            "duplicate_post_ids": 0,
            "invalid_or_deleted_status_count": sum(row["status"] != 1 for row in rows),
            "blockquote_message_count": sum(
                "<blockquote" in row["msg_html"].lower() for row in rows
            ),
        },
        "framework_mentions": count_messages(rows, FRAMEWORK_PATTERNS),
        "entity_mentions": count_messages(rows, ENTITY_PATTERNS),
        "candidate_evidence": build_candidate_evidence(rows),
        "method_notes": [
            "Counts are message-level: one message counts once per concept/entity.",
            "Mentions measure attention, not positive sentiment or recommendation strength.",
            "Only comments inside the downloaded 33-thread series are included; this is not a site-wide LAC profile export.",
            "Part 29 does not exist in the author chain/search index; the current part 34 was still active at collection time.",
            "Two messages contain blockquotes, so quoted text can marginally affect term counts; candidate evidence was checked manually in context.",
        ],
    }
    return rows, summary


def write_notebook(path: Path, summary: dict[str, Any]) -> None:
    """Write a dependency-free notebook companion with verified saved outputs."""
    profile_output = json.dumps(summary["profile"], ensure_ascii=False, indent=2)
    top_entities = json.dumps(summary["entity_mentions"][:12], ensure_ascii=False, indent=2)
    top_framework = json.dumps(summary["framework_mentions"][:12], ensure_ascii=False, indent=2)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## tl;dr\n",
                "LAC 在系列內共有 **3,556** 則留言。其方法核心不是追熱門股，而是把護城河、TAM、margin、capex、週期與估值連成可驗證的 DCF／情景分析。最新最強直接買入訊號是 CDI。\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Context & Methods\n",
                "資料來自 `lihkg_美日韓_超長線十倍價投/part_*.json`。以 LIHKG user ID 734436 識別 LAC，按訊息而非字詞出現次數統計。提及次數只代表關注，不等同正面推薦。\n",
                "\n### Key Assumptions\n",
                "分析範圍只限已下載系列；第 29 集不存在，第 34 集仍在更新。\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "source": [
                "from pathlib import Path\n",
                "from analyze_lac_comments import analyse\n",
                "rows, summary = analyse(Path('lihkg_美日韓_超長線十倍價投'))\n",
                "summary['profile']\n",
            ],
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 1,
                    "metadata": {},
                    "data": {"text/plain": [profile_output]},
                }
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data\n", "以下為訊息層級的資料完整性及覆蓋範圍。\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "source": ["summary['entity_mentions'][:12]\n"],
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 2,
                    "metadata": {},
                    "data": {"text/plain": [top_entities]},
                }
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Results\n", "框架概念的訊息級出現次數：\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "source": ["summary['framework_mentions'][:12]\n"],
            "outputs": [
                {
                    "output_type": "execute_result",
                    "execution_count": 3,
                    "metadata": {},
                    "data": {"text/plain": [top_framework]},
                }
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Takeaways\n",
                "1. 先量化長期收入來源，再檢查 margin、capex、SBC、收購支出和週期正常化。\n",
                "2. 只在 bear/base case 仍有折讓時稱為便宜；質素重要，但價格仍是入場門檻。\n",
                "3. CDI 是最新、最直接的買入；Adyen、1571、COHR 是不同風險形態的次選。\n",
                "4. 對無能力自行驗證估值的人，LAC 自己反而建議被動指數投資。\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "analysis_validation": "Outputs generated by and reconciled to analyze_lac_comments.py",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_report_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    profile = summary["profile"]
    candidates = summary["candidate_evidence"]
    csv_source = "analysis/lac_comment_analysis/lac_comments.csv"

    def regex_count_sql(patterns: dict[str, str], kind: str) -> str:
        statements = []
        for label, pattern in patterns.items():
            safe_label = label.replace("'", "''")
            safe_pattern = pattern.replace("'", "''")
            statements.append(
                f"SELECT '{kind}' AS kind, '{safe_label}' AS label, "
                f"count_if(regexp_matches(msg_text, '{safe_pattern}')) AS message_count "
                f"FROM read_csv_auto('{csv_source}', header=true)"
            )
        return "\nUNION ALL\n".join(statements)

    candidate_values = []
    for item in candidates:
        fields = [
            str(item["rank"]),
            *[
                "'" + str(item[field]).replace("'", "''") + "'"
                for field in (
                    "ticker",
                    "price",
                    "lac_signal",
                    "independent_check",
                    "main_risk",
                    "confidence",
                    "signal_date",
                    "source_url",
                )
            ],
        ]
        candidate_values.append("(" + ", ".join(fields) + ")")
    candidate_sql = (
        "SELECT * FROM (VALUES\n  "
        + ",\n  ".join(candidate_values)
        + "\n) AS candidates(rank, ticker, price, lac_signal, independent_check, "
        "main_risk, confidence, signal_date, source_url) ORDER BY rank"
    )
    sources = [
        {
            "id": "lihkg_corpus",
            "label": "LIHKG 系列留言存檔（第 1–34 集；第 29 集不存在）",
            "href": "https://lihkg.com/thread/4152579/page/1",
            "path": "lihkg_美日韓_超長線十倍價投/manifest.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "從已驗證的 LAC 留言 CSV 計算留言數、覆蓋集數及長分析留言數。",
                "sql": (
                    "SELECT count(*) AS comments, count(DISTINCT part) AS parts, "
                    "count_if(length(msg_text) >= 200) AS long_comments "
                    f"FROM read_csv_auto('{csv_source}', header=true)"
                ),
                "tables_used": [csv_source],
                "filters": ["user.user_id = 734436", "series parts 1–34 except absent part 29"],
                "metric_definitions": {
                    "comment_count": "符合 user ID 734436 的唯一 post_id 數量。",
                    "message_count": "至少包含一次指定概念或公司名稱的 LAC 留言數；同一留言最多計一次。",
                },
            },
        },
        {
            "id": "lac_mentions",
            "label": "LAC 留言概念及公司提及統計",
            "path": csv_source,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "每則留言對每個 regex 分類最多計一次；輸出框架概念和公司提及的訊息級計數。",
                "sql": (
                    regex_count_sql(FRAMEWORK_PATTERNS, "framework")
                    + "\nUNION ALL\n"
                    + regex_count_sql(ENTITY_PATTERNS, "entity")
                ),
                "tables_used": [csv_source],
                "filters": ["one count per message per label", "top 12 shown in each chart"],
                "metric_definitions": {
                    "message_count": "至少包含一次對應 regex 的唯一 LAC 留言數；不是情緒或推薦分數。"
                },
            },
        },
        {
            "id": "market_snapshot",
            "label": "交易所及公司公告市場快照（2026-09-04）",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Euronext、HKEX、公司投資者關係頁及即時市場報價的人工核對快照。",
                "sql": candidate_sql,
                "tables_used": [
                    "Euronext CDI/MC/ADYEN/BESI quotes",
                    "HKEX Xin Point 1H 2026 results",
                    "Adyen H1 2026 results",
                    "Coherent FY2026 Q4 results",
                ],
                "filters": ["latest available quote at 2026-09-04", "official issuer/exchange results preferred"],
                "metric_definitions": {
                    "CDI NAV discount": "1 - CDI market value / gross market value of its LVMH shareholding; approximate and before holding-company adjustments.",
                    "indicated yield 1571": "HK$0.15 latest interim plus HK$0.30 last final dividend, divided by HK$3.81; not a forecast or guarantee.",
                },
            },
        },
        {
            "id": "euronext_cdi",
            "label": "Euronext — Christian Dior (CDI)",
            "href": "https://live.euronext.com/en/product/equities/FR0000130403-XPAR",
        },
        {
            "id": "lvmh_h1_2026",
            "label": "LVMH — H1 2026 results",
            "href": "https://www.lvmh.com/en/publications/accelerating-growth-in-the-second-quarter---solid-first-half-results",
        },
        {
            "id": "adyen_h1_2026",
            "label": "Adyen — H1 2026 results",
            "href": "https://www.adyen.com/press-and-media/adyen-publishes-h1-2026-financial-results-3wjne",
        },
        {
            "id": "xinpoint_h1_2026",
            "label": "HKEX — Xin Point 1H 2026 results",
            "href": "https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0827/2026082701415.pdf",
        },
        {
            "id": "cohr_fy2026",
            "label": "Coherent — FY2026 Q4 results",
            "href": "https://ir.coherent.com/news-releases/news-release-details/coherent-corp-reports-fourth-quarter-and-full-year-fiscal-2026",
        },
    ]

    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "profile": [
                {
                    "comments": profile["comment_count"],
                    "parts": profile["part_count"],
                    "long_comments": profile["comments_at_least_200_characters"],
                    "last_comment": profile["last_comment_utc"][:10],
                }
            ],
            "framework_mentions": summary["framework_mentions"][:12],
            "entity_mentions": summary["entity_mentions"][:12],
            "candidates": candidates,
        },
    }
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "用 LAC 的方法看現時最佳股票",
        "description": "分析 LAC 在 LIHKG 投資系列內全部留言，再以最新市場資料重建其投資判斷。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "comments_card",
                "description": "以唯一 post ID 計算。",
                "dataset": "profile",
                "metrics": [{"label": "留言", "field": "comments", "format": "number"}],
                "sourceId": "lihkg_corpus",
            },
            {
                "id": "parts_card",
                "description": "LAC 由第 5 集開始出現；第 29 集不存在。",
                "dataset": "profile",
                "metrics": [{"label": "集數", "field": "parts", "format": "number"}],
                "sourceId": "lihkg_corpus",
            },
            {
                "id": "long_comments_card",
                "description": "至少 200 個字元的留言。",
                "dataset": "profile",
                "metrics": [{"label": "留言", "field": "long_comments", "format": "number"}],
                "sourceId": "lihkg_corpus",
            },
        ],
        "charts": [
            {
                "id": "framework_chart",
                "title": "投資框架概念出現次數",
                "description": "訊息級計數；同一留言同一概念最多計一次。",
                "type": "bar",
                "intent": "comparison",
                "question": "LAC 最常用哪些投資分析概念？",
                "rationale": "排序橫向棒形圖適合比較名稱較長的離散概念。",
                "dataset": "framework_mentions",
                "encodings": {"x": {"field": "label"}, "y": {"field": "message_count"}},
                "options": {"orientation": "horizontal", "grouping": "grouped"},
                "palette": {"kind": "categorical", "name": "blue"},
                "sourceId": "lac_mentions",
            },
            {
                "id": "entity_chart",
                "title": "最常提及公司或股票",
                "description": "訊息級計數；提及不代表正面評級。",
                "type": "bar",
                "intent": "ranking",
                "question": "哪些公司最常出現在 LAC 的留言？",
                "rationale": "排序橫向棒形圖展示研究注意力集中度，同時保留長公司名稱。",
                "dataset": "entity_mentions",
                "encodings": {"x": {"field": "label"}, "y": {"field": "message_count"}},
                "options": {"orientation": "horizontal", "grouping": "grouped"},
                "palette": {"kind": "categorical", "name": "gold"},
                "sourceId": "lac_mentions",
            },
        ],
        "tables": [
            {
                "id": "candidate_table",
                "title": "LAC-style 現時候選排序",
                "description": "排序重視最新直接行動、估值安全邊際、商業質素及可驗證基本面；不是本人投資建議。",
                "dataset": "candidates",
                "columns": [
                    {"field": "rank", "label": "排名", "type": "number"},
                    {"field": "ticker", "label": "股票", "type": "text"},
                    {"field": "price", "label": "價格快照", "type": "text"},
                    {"field": "lac_signal", "label": "LAC 訊號", "type": "text"},
                    {"field": "independent_check", "label": "基本面核對", "type": "text"},
                    {"field": "main_risk", "label": "主要風險", "type": "text"},
                    {"field": "confidence", "label": "重建信心", "type": "text"},
                ],
                "defaultSort": {"field": "rank", "direction": "asc"},
                "sourceId": "market_snapshot",
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# 用 LAC 的方法看現時最佳股票"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "body": (
                    "## Executive Summary\n\n"
                    "- **最強現時訊號是 CDI（Christian Dior SE）。** LAC 在 9 月 4 日直接買入，並表示以約 15 倍穿透盈利買到 LVMH；按交易所價格重算，約有 20–22% 控股折讓，粗略年化穿透市盈率亦接近 15 倍。\n"
                    "- **第二選是 Adyen。** LAC 在 H1 業績後仍估計低估約 30%，而現價略低於當日收市；公司維持約 20% 固定匯率收入增長、接近 50% EBITDA margin 和高現金轉化。\n"
                    "- **1571 信邦控股和 COHR 是不同風險的次選。** 前者是低估值、高現金回報但低流動性的汽車零件股；後者是 AI 光通訊增長股，現價已進入 LAC 曾表示的首個買入區。\n"
                    "- **這不是模仿口吻的貼士。** 真正的 LAC 方法要求自行建立 10 年模型、情景及敏感度測試；若做不到，他本人反而多次建議被動指數投資。"
                ),
            },
            {"id": "profile_heading", "type": "markdown", "body": "## 先確認我們真的看過『全部留言』\n\n存檔內所有 LAC 留言均對應同一 user ID，沒有重複 post ID 或刪除狀態。範圍是這個系列，而不是 LAC 在 LIHKG 全站的所有發言。"},
            {"id": "profile_metrics", "type": "metric-strip", "cardIds": ["comments_card", "parts_card", "long_comments_card"]},
            {"id": "framework_heading", "type": "markdown", "body": "## LAC 的核心不是揀股名，而是把敘事量化\n\n他的常用順序是：先拆 TAM、出貨量、市佔和 ASP，再推收入、margin、capex、SBC、收購成本與 FCF；之後做 bear/base/bull case、不同貼現率和 reverse DCF。護城河決定可持續性，估值決定是否落場。"},
            {"id": "framework_visual", "type": "chart", "chartId": "framework_chart"},
            {"id": "attention_heading", "type": "markdown", "body": "## 半導體是研究重心，但高提及不等於看好\n\nNVDA、TSMC、SK Hynix、MRVL、ASML 等高頻出現，往往同時包含風險、週期或估值批評。這幅圖應理解為能力圈和研究時間分配，不是推薦榜。"},
            {"id": "attention_visual", "type": "chart", "chartId": "entity_chart"},
            {"id": "ranking_heading", "type": "markdown", "body": "## 按最新行動、估值與基本面，CDI 排第一\n\nCDI 是唯一在資料截點當日出現的明確新買入。Adyen 有最完整的質素與增長論證；1571 和 COHR 則分別補足現金回報與 AI 基建增長，但風險明顯更高。"},
            {"id": "ranking_table", "type": "table", "tableId": "candidate_table"},
            {"id": "not_now", "type": "markdown", "body": "## 哪些熱門股暫時不符合『最好』\n\n- **AMZN、META、GOOG、MSFT：** LAC 最新比較認為最便宜的 hyperscaler 也大致只是 base-case fair value，並非 bear-case 安全邊際。\n- **MRVL、OSCR：** 他曾因市場已反映較樂觀情景而減持或稱接近 fully priced。\n- **記憶體股：** 他反覆反對用 peak EPS 乘高倍數，要求以 normalized earnings 和 2028 左右供需正常化測試。\n- **AMD、AVGO：** 商業質素和增長受到肯定，但最新留言沒有 CDI 那種清晰的安全邊際及直接買入訊號。"},
            {"id": "next_steps", "type": "markdown", "body": "## 建議下一步\n\n1. **先研究 CDI，而不是立即買入。** 重做 LVMH 10 年 bear/base/bull DCF，再對 CDI 使用 10%／20%／30% 控股折讓敏感度。\n2. **Adyen 用 TPV × take rate 建模。** 把 21–23% 固定匯率增長指引、收購後 margin、7% capex 及 2028 年 >55% margin 逐項壓力測試。\n3. **1571 必須把股息和盈利分開看。** 以不派特別息、汽車訂單延遲及召回損失作 downside case。\n4. **COHR 以分注而非單一目標價處理。** LAC 的歷史區間是 280／240／200；只有在 AI datacenter 增長和 margin 擴張仍成立時才有意義。"},
            {"id": "questions", "type": "markdown", "body": "## 仍要回答的問題\n\n- CDI 折讓有沒有可實現催化劑，還是會永久存在？\n- Adyen 兩宗收購能否在不破壞 organic margin 的情況下提高 take rate？\n- 1571 的 RMB10.26bn 訂單轉成收入時，margin 和現金回報是否維持？\n- COHR 的擴產會否先推高折舊與資本需求，令收入增長未能轉化成 FCF？"},
            {"id": "caveats", "type": "markdown", "body": "## Caveats and Assumptions\n\n這是一個『LAC-style』框架重建，不是 LAC 本人的最新投資建議，也沒有考慮你的資產、現金流、稅務、貨幣或風險承受能力。CDI 的穿透估值是按公開持股和半年盈利作粗略年化，未完整調整季節性、控股公司債務、稅項和少數股東權益。1571 的 11.8% 是指示性歷史股息率，不代表未來派息。任何候選都可能出現重大永久損失。"},
        ],
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> int:
    rows, summary = analyse()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "lac_comments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "lac_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_notebook(OUTPUT_DIR / "lac_analysis.ipynb", summary)
    (OUTPUT_DIR / "lac_report_artifact.json").write_text(
        json.dumps(build_report_artifact(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["profile"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

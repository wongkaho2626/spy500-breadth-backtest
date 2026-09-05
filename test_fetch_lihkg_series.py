import json
from argparse import Namespace

import pytest

from fetch_lihkg_series import (
    ScrapeError,
    discover_series,
    message_to_text,
    normalise_thread,
    parse_part_number,
    parse_thread_id,
    response_payload,
    series_part_number,
)


def test_parse_thread_id_accepts_full_thread_urls():
    assert parse_thread_id("https://lihkg.com/thread/4152579/page/1") == 4152579
    assert parse_thread_id("https://www.lihkg.com/thread/123/") == 123


def test_parse_thread_id_rejects_unrelated_urls():
    with pytest.raises(ValueError):
        parse_thread_id("https://example.com/thread/4152579/page/1")


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("美日韓 超長線十倍價投 [34]", 34),
        ("美日韓 超長線十倍價投 【3】", 3),
        ("美日韓 超長線十倍價投 (1)", 1),
        ("no part", None),
    ],
)
def test_parse_part_number(title, expected):
    assert parse_part_number(title) == expected


def test_series_part_number_maps_unnumbered_legacy_title_to_one():
    assert series_part_number("賭Nio一年半內執笠") == 1


def test_message_to_text_preserves_links_images_and_line_breaks():
    value = message_to_text(
        'first<br><a href="/thread/1/page/1">previous</a>'
        '<p>chart <img src="/assets/chart.png" alt="chart"></p>'
    )
    assert value == (
        "first\nprevious (https://lihkg.com/thread/1/page/1)\n"
        "chart [chart: https://lihkg.com/assets/chart.png]"
    )


def test_response_payload_validates_successful_api_document():
    result = {
        "status": 200,
        "body": json.dumps({"success": 1, "response": {"thread_id": 9}}),
    }
    assert response_payload(result, 9, 1) == {"thread_id": 9}


def test_response_payload_rejects_http_errors():
    with pytest.raises(ScrapeError, match="HTTP 429"):
        response_payload({"status": 429, "body": "slow down"}, 9, 1)


def test_normalise_thread_orders_posts_and_adds_plain_text():
    page = {
        "thread_id": 7,
        "title": "美日韓 超長線十倍價投 [1]",
        "page": 1,
        "total_page": 1,
        "item_data": [
            {"msg_num": 2, "msg": "second", "reply_time": 10},
            {"msg_num": 1, "msg": "first<br>line", "reply_time": 5},
        ],
    }
    result = normalise_thread(1, [page])
    assert [post["msg_num"] for post in result["posts"]] == [1, 2]
    assert result["posts"][0]["msg_text"] == "first\nline"
    assert result["posts"][0]["reply_time_utc"] == "1970-01-01T00:00:05+00:00"


class FakeSession:
    def __init__(self):
        self.pages = {
            (300, 1): {
                "thread_id": 300,
                "title": "美日韓 超長線十倍價投 [3]",
                "parent_thread_id": 200,
                "user_id": 77,
                "total_page": 2,
                "item_data": [],
            },
            (200, 1): {
                "thread_id": 200,
                "title": "美日韓 超長線十倍價投 [2]",
                "parent_thread_id": 100,
                "user_id": 77,
                "total_page": 3,
                "item_data": [],
            },
            (100, 1): {
                "thread_id": 100,
                "title": "美日韓 超長線十倍價投 [1]",
                "parent_thread_id": 99,
                "user_id": 77,
                "total_page": 4,
                "item_data": [],
            },
        }

    def fetch_api_pages(self, requests):
        return [
            {
                "status": 200,
                "body": json.dumps({"success": 1, "response": self.pages[request]}),
            }
            for request in requests
        ]

    def fetch_api_urls(self, urls):
        items = [
            {
                "thread_id": payload["thread_id"],
                "title": payload["title"],
                "user_id": payload["user_id"],
                "no_of_reply": 1001,
                "total_page": payload["total_page"],
                "create_time": payload["thread_id"],
            }
            for payload in self.pages.values()
        ]
        body = json.dumps(
            {"success": 1, "response": {"is_pagination": False, "items": items}}
        )
        return [{"status": 200, "body": body} for _ in urls]


def test_discover_series_follows_parent_thread_ids():
    ids, cache, missing = discover_series(FakeSession(), 300, 1, 3)
    assert ids == {3: 300, 2: 200, 1: 100}
    assert cache[(300, 1)]["title"].endswith("[3]")
    assert (100, 1) not in cache
    assert missing == []


def test_discover_series_records_skipped_numbers():
    session = FakeSession()
    session.pages[(300, 1)]["parent_thread_id"] = 100
    original_search = session.fetch_api_urls

    def search_without_part_two(urls):
        results = original_search(urls)
        for result in results:
            document = json.loads(result["body"])
            document["response"]["items"] = [
                item
                for item in document["response"]["items"]
                if not item["title"].endswith("[2]")
            ]
            result["body"] = json.dumps(document)
        return results

    session.fetch_api_urls = search_without_part_two
    ids, _, missing = discover_series(session, 300, 1, 3)
    assert ids == {3: 300, 1: 100}
    assert missing == [2]

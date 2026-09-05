from analyze_lac_comments import analyse, build_report_artifact


def test_lac_archive_profile_is_complete_and_unique():
    rows, summary = analyse()
    profile = summary["profile"]
    assert len(rows) == profile["comment_count"] == 3556
    assert profile["duplicate_post_ids"] == 0
    assert profile["invalid_or_deleted_status_count"] == 0
    assert profile["part_count"] == 29
    assert all(row["user_id"] == 734436 for row in rows)


def test_candidate_ranking_uses_latest_direct_signals():
    _, summary = analyse()
    candidates = summary["candidate_evidence"]
    assert [item["ticker"] for item in candidates] == [
        "CDI (Paris)",
        "ADYEN (Amsterdam)",
        "1571 (Hong Kong)",
        "COHR (NYSE)",
    ]
    assert candidates[0]["signal_date"] == "2026-09-04"
    assert all(item["source_url"].startswith("https://lihkg.com/thread/") for item in candidates)


def test_report_artifact_has_bounded_snapshot_and_required_reading_path():
    _, summary = analyse()
    artifact = build_report_artifact(summary)
    assert artifact["surface"] == "report"
    assert len(artifact["snapshot"]["datasets"]) == 4
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# ")
    assert artifact["manifest"]["blocks"][1]["body"].startswith("## Executive Summary")
    assert len(artifact["manifest"]["charts"]) >= 1

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DATA_API_BASE = os.environ.get("DATA_API_BASE", "http://129.114.27.10:30800").rstrip("/")

MEETING_MANIFEST_PATH = Path(
    os.environ.get("MEETING_MANIFEST_PATH", "data/meeting_manifest.jsonl")
)

OUTPUT_DIR = Path(os.environ.get("RETRAIN_DATA_DIR", "data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "retraining_ready_train.jsonl"
VAL_PATH = OUTPUT_DIR / "retraining_ready_val.jsonl"
TEST_PATH = OUTPUT_DIR / "retraining_ready_test.jsonl"
STATS_PATH = OUTPUT_DIR / "retraining_stats.json"

SEED = int(os.environ.get("DATASET_SPLIT_SEED", "42"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.1"))

MIN_RATING = int(os.environ.get("MIN_RATING", "4"))
REQUIRE_APPROVED = os.environ.get("REQUIRE_APPROVED", "true").lower() == "true"
REQUIRE_EDITED_SUMMARY = os.environ.get("REQUIRE_EDITED_SUMMARY", "true").lower() == "true"
MIN_SUMMARY_CHARS = int(os.environ.get("MIN_SUMMARY_CHARS", "10"))

REQUEST_TIMEOUT = (10, 120)


def make_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return f"{DATA_API_BASE}{path}"


def get_json(path: str) -> Any:
    url = make_url(path)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)

    if not resp.ok:
        raise RuntimeError(
            f"GET failed: {url} | status={resp.status_code} | body={resp.text}"
        )

    try:
        return resp.json()
    except Exception:
        return resp.text


def normalize_payload_to_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]

    return {"raw_text": payload}


def normalize_payload_to_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("items", "results", "data", "reviews", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        return [payload]

    return []


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "approved", "approve"}


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize_action_items(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]


def get_review_id(review: Dict[str, Any]) -> str:
    return str(
        review.get("review_id")
        or review.get("id")
        or review.get("reviewId")
        or ""
    ).strip()


def get_meeting_id_from_review(review: Dict[str, Any]) -> str:
    meeting_id = str(
        review.get("meeting_id")
        or review.get("meetingId")
        or review.get("meeting_uuid")
        or ""
    ).strip()

    if meeting_id:
        return meeting_id

    meeting_obj = review.get("meeting")
    if isinstance(meeting_obj, dict):
        return str(
            meeting_obj.get("meeting_id")
            or meeting_obj.get("id")
            or meeting_obj.get("meetingId")
            or ""
        ).strip()

    return ""


def read_meeting_ids_from_env() -> List[str]:
    raw = os.environ.get("MEETING_IDS", "").strip()

    if not raw:
        return []

    meeting_ids: List[str] = []
    seen = set()

    for part in raw.replace("\n", ",").replace(" ", ",").split(","):
        meeting_id = part.strip()
        if meeting_id and meeting_id not in seen:
            seen.add(meeting_id)
            meeting_ids.append(meeting_id)

    return meeting_ids


def read_meeting_ids_from_manifest(path: Path) -> List[str]:
    if not path.exists():
        print(f"[WARN] Meeting manifest not found: {path}")
        return []

    meeting_ids: List[str] = []
    seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                print(f"[WARN] Skipping invalid manifest line: {line}")
                continue

            meeting_id = str(
                row.get("meeting_id")
                or row.get("meetingId")
                or row.get("id")
                or ""
            ).strip()

            if meeting_id and meeting_id not in seen:
                seen.add(meeting_id)
                meeting_ids.append(meeting_id)

    return meeting_ids


def get_meeting_ids() -> Tuple[List[str], str]:
    env_meeting_ids = read_meeting_ids_from_env()

    if env_meeting_ids:
        return env_meeting_ids, "MEETING_IDS"

    manifest_meeting_ids = read_meeting_ids_from_manifest(MEETING_MANIFEST_PATH)

    if manifest_meeting_ids:
        return manifest_meeting_ids, str(MEETING_MANIFEST_PATH)

    return [], str(MEETING_MANIFEST_PATH)


def get_reviews_by_meeting(meeting_id: str) -> List[Dict[str, Any]]:
    payload = get_json(f"/reviews/by_meeting/{meeting_id}")
    return normalize_payload_to_list(payload)


def get_review_by_id(review_id: str) -> Dict[str, Any]:
    payload = get_json(f"/reviews/{review_id}")
    return normalize_payload_to_dict(payload)


def get_reviews_from_review_ids() -> Tuple[List[Dict[str, Any]], str]:
    raw = os.environ.get("REVIEW_IDS", "").strip()

    if not raw:
        return [], ""

    review_ids: List[str] = []
    seen = set()

    for part in raw.replace("\n", ",").replace(" ", ",").split(","):
        review_id = part.strip()
        if review_id and review_id not in seen:
            seen.add(review_id)
            review_ids.append(review_id)

    reviews: List[Dict[str, Any]] = []
    errors: List[str] = []

    for review_id in review_ids:
        try:
            review = get_review_by_id(review_id)
            reviews.append(review)
        except Exception as e:
            errors.append(f"{review_id}: {e}")
            print(f"[WARN] Could not fetch review_id={review_id}: {e}")

    if errors:
        print("[WARN] Some REVIEW_IDS could not be loaded:")
        for err in errors:
            print(f"[WARN] {err}")

    return reviews, "REVIEW_IDS"


def get_meeting(meeting_id: str) -> Dict[str, Any]:
    try:
        return normalize_payload_to_dict(get_json(f"/meetings/{meeting_id}"))
    except Exception as e:
        print(f"[WARN] Could not fetch meeting {meeting_id}: {e}")
        return {}


def get_summary_by_meeting(meeting_id: str) -> Dict[str, Any]:
    payload = get_json(f"/summaries/by_meeting/{meeting_id}")
    return normalize_payload_to_dict(payload)


def extract_original_summary(
    summary_record: Optional[Dict[str, Any]],
    review: Optional[Dict[str, Any]] = None,
) -> str:
    summary_record = summary_record or {}
    review = review or {}

    return str(
        summary_record.get("summary_text")
        or summary_record.get("summary")
        or summary_record.get("text")
        or summary_record.get("content")
        or summary_record.get("raw_text")
        or review.get("original_summary")
        or review.get("summary")
        or ""
    ).strip()


def review_fingerprint(meeting_id: str, review: Dict[str, Any]) -> str:
    payload = {
        "meeting_id": meeting_id,
        "review_id": get_review_id(review),
        "reviewer_id": str(review.get("reviewer_id", "") or ""),
        "rating": to_int(review.get("rating", 0), 0),
        "approved": to_bool(review.get("approved", False)),
        "edited_summary": str(review.get("edited_summary", "") or "").strip(),
        "edited_action_items": normalize_action_items(review.get("edited_action_items")),
        "review_notes": str(review.get("review_notes", "") or "").strip(),
        "correction_label": str(review.get("correction_label", "") or "").strip(),
    }

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_example(
    meeting_id: str,
    review: Dict[str, Any],
    summary_record: Optional[Dict[str, Any]] = None,
    meeting_record: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    approved = to_bool(review.get("approved", False))
    rating = to_int(review.get("rating", 0), 0)
    edited_summary = str(review.get("edited_summary", "") or "").strip()
    correction_label = str(review.get("correction_label", "") or "").strip()
    reviewer_id = str(review.get("reviewer_id", "") or "").strip()
    review_notes = str(review.get("review_notes", "") or "").strip()
    edited_action_items = normalize_action_items(review.get("edited_action_items"))

    if REQUIRE_APPROVED and not approved:
        return None

    if rating < MIN_RATING:
        return None

    if REQUIRE_EDITED_SUMMARY and not edited_summary:
        return None

    if len(edited_summary) < MIN_SUMMARY_CHARS:
        return None

    original_summary = extract_original_summary(summary_record, review)

    if not original_summary:
        return None

    audio_object_key = ""
    if meeting_record:
        audio_object_key = str(meeting_record.get("audio_object_key", "") or "").strip()

    fp = review_fingerprint(meeting_id, review)

    return {
        "meeting_id": meeting_id,

        # Important:
        # The field name is kept as input_transcript so train.py and config.yaml
        # do not need to change. However, the value is the original generated summary.
        # This makes the retraining task:
        # original_summary -> edited_summary
        "input_transcript": original_summary,

        "target_summary": edited_summary,
        "original_summary": original_summary,
        "rating": rating,
        "approved": approved,
        "correction_label": correction_label,
        "reviewer_id": reviewer_id,
        "review_notes": review_notes,
        "edited_action_items": edited_action_items,
        "audio_object_key": audio_object_key,
        "source": "reviews_api_summary_correction",
        "source_review_id": get_review_id(review),
        "source_review_fingerprint": fp,
    }


def split_examples(
    examples: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for ex in examples:
        grouped.setdefault(ex["meeting_id"], []).append(ex)

    meeting_ids = list(grouped.keys())
    rng = random.Random(SEED)
    rng.shuffle(meeting_ids)

    # Preferred: meeting-level split.
    if len(meeting_ids) >= 2:
        n = len(meeting_ids)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))

        if n >= 3:
            if TEST_RATIO > 0:
                n_test = max(1, n_test)
            if VAL_RATIO > 0:
                n_val = max(1, n_val)

        if n_test + n_val >= n:
            n_test = 0
            n_val = max(1, min(n - 1, n_val))

        test_ids = set(meeting_ids[:n_test])
        val_ids = set(meeting_ids[n_test:n_test + n_val])
        train_ids = set(meeting_ids[n_test + n_val:])

        if not train_ids and meeting_ids:
            last_id = meeting_ids[-1]
            train_ids.add(last_id)
            val_ids.discard(last_id)
            test_ids.discard(last_id)

        train_rows, val_rows, test_rows = [], [], []

        for meeting_id, rows in grouped.items():
            if meeting_id in test_ids:
                test_rows.extend(rows)
            elif meeting_id in val_ids:
                val_rows.extend(rows)
            else:
                train_rows.extend(rows)

        # Safety fallback if validation accidentally becomes empty.
        if not val_rows and train_rows:
            val_rows = train_rows[:1]

        return train_rows, val_rows, test_rows, "meeting_level"

    # Fallback for demo/testing when only one meeting has feedback.
    rows = examples[:]
    rng.shuffle(rows)

    if len(rows) == 1:
        return rows, rows, [], "single_example_fallback"

    val_count = max(1, int(round(len(rows) * VAL_RATIO)))

    if val_count >= len(rows):
        val_count = 1

    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    if not train_rows:
        train_rows = rows[1:]
        val_rows = rows[:1]

    return train_rows, val_rows, [], "single_meeting_example_fallback"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_fingerprint(examples: List[Dict[str, Any]]) -> str:
    fps = sorted(str(x.get("source_review_fingerprint", "")) for x in examples)
    raw = "\n".join(fps)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def collect_examples_from_meeting_ids(
    meeting_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    skipped_meetings = 0
    skipped_reviews = 0
    total_reviews_seen = 0
    errors: List[str] = []

    meeting_cache: Dict[str, Dict[str, Any]] = {}
    summary_cache: Dict[str, Dict[str, Any]] = {}

    for meeting_id in meeting_ids:
        try:
            if meeting_id not in meeting_cache:
                meeting_cache[meeting_id] = get_meeting(meeting_id)

            if meeting_id not in summary_cache:
                summary_cache[meeting_id] = get_summary_by_meeting(meeting_id)

            reviews = get_reviews_by_meeting(meeting_id)
            total_reviews_seen += len(reviews)

            for review in reviews:
                ex = build_example(
                    meeting_id=meeting_id,
                    review=review,
                    summary_record=summary_cache[meeting_id],
                    meeting_record=meeting_cache[meeting_id],
                )

                if ex is not None:
                    examples.append(ex)
                else:
                    skipped_reviews += 1

        except Exception as e:
            skipped_meetings += 1
            errors.append(f"meeting_id={meeting_id}: {e}")
            print(f"[WARN] Skipping meeting_id={meeting_id} due to error: {e}")

    stats = {
        "skipped_meetings": skipped_meetings,
        "skipped_reviews": skipped_reviews,
        "total_reviews_seen": total_reviews_seen,
        "errors": errors,
    }

    return examples, stats


def collect_examples_from_review_ids(
    reviews: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    skipped_reviews = 0
    reviews_without_meeting_id = 0
    errors: List[str] = []

    meeting_cache: Dict[str, Dict[str, Any]] = {}
    summary_cache: Dict[str, Dict[str, Any]] = {}

    for review in reviews:
        meeting_id = get_meeting_id_from_review(review)

        if not meeting_id:
            reviews_without_meeting_id += 1
            skipped_reviews += 1
            errors.append(f"review_id={get_review_id(review)} has no meeting_id")
            continue

        try:
            if meeting_id not in meeting_cache:
                meeting_cache[meeting_id] = get_meeting(meeting_id)

            if meeting_id not in summary_cache:
                summary_cache[meeting_id] = get_summary_by_meeting(meeting_id)

            ex = build_example(
                meeting_id=meeting_id,
                review=review,
                summary_record=summary_cache[meeting_id],
                meeting_record=meeting_cache[meeting_id],
            )

            if ex is not None:
                examples.append(ex)
            else:
                skipped_reviews += 1

        except Exception as e:
            skipped_reviews += 1
            errors.append(
                f"meeting_id={meeting_id}, review_id={get_review_id(review)}: {e}"
            )
            print(
                f"[WARN] Skipping review_id={get_review_id(review)} "
                f"meeting_id={meeting_id} due to error: {e}"
            )

    stats = {
        "skipped_reviews": skipped_reviews,
        "reviews_without_meeting_id": reviews_without_meeting_id,
        "errors": errors,
    }

    return examples, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty output files instead of failing when no eligible examples exist.",
    )
    args = parser.parse_args()

    meeting_ids, meeting_id_source = get_meeting_ids()
    review_id_reviews, review_id_source = get_reviews_from_review_ids()

    print(f"[INFO] Loaded {len(meeting_ids)} meeting_id(s) from {meeting_id_source}")

    if review_id_source:
        print(f"[INFO] Loaded {len(review_id_reviews)} review(s) from {review_id_source}")

    examples: List[Dict[str, Any]] = []

    meeting_stats = {
        "skipped_meetings": 0,
        "skipped_reviews": 0,
        "total_reviews_seen": 0,
        "errors": [],
    }

    review_id_stats = {
        "skipped_reviews": 0,
        "reviews_without_meeting_id": 0,
        "errors": [],
    }

    if meeting_ids:
        meeting_examples, meeting_stats = collect_examples_from_meeting_ids(meeting_ids)
        examples.extend(meeting_examples)

    if review_id_reviews:
        review_id_examples, review_id_stats = collect_examples_from_review_ids(
            review_id_reviews
        )
        examples.extend(review_id_examples)

    # De-duplicate by review fingerprint.
    deduped: Dict[str, Dict[str, Any]] = {}

    for ex in examples:
        deduped[ex["source_review_fingerprint"]] = ex

    examples = list(deduped.values())

    if not examples and not args.write_empty:
        raise RuntimeError(
            "No eligible retraining examples were built from API review data. "
            "Check meeting IDs, review IDs, /reviews/by_meeting/{meeting_id}, "
            "/reviews/{review_id}, /summaries/by_meeting/{meeting_id}, ratings, "
            "approved flag, and edited_summary."
        )

    if examples:
        train_rows, val_rows, test_rows, split_mode = split_examples(examples)
    else:
        train_rows, val_rows, test_rows, split_mode = [], [], [], "empty"

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)
    write_jsonl(TEST_PATH, test_rows)

    all_errors: List[str] = []
    all_errors.extend(meeting_stats.get("errors", []))
    all_errors.extend(review_id_stats.get("errors", []))

    stats = {
        "task_type": "summary_correction",
        "model_input": "original_summary",
        "model_target": "edited_summary",
        "note": (
            "The output JSONL keeps the field name input_transcript for compatibility "
            "with train.py/config.yaml, but the value is the original generated summary."
        ),
        "meeting_id_source": meeting_id_source,
        "meeting_ids_loaded": len(meeting_ids),
        "review_id_source": review_id_source,
        "review_ids_loaded": len(review_id_reviews),
        "total_reviews_seen": meeting_stats.get("total_reviews_seen", 0)
        + len(review_id_reviews),
        "skipped_meetings": meeting_stats.get("skipped_meetings", 0),
        "skipped_reviews": meeting_stats.get("skipped_reviews", 0)
        + review_id_stats.get("skipped_reviews", 0),
        "reviews_without_meeting_id": review_id_stats.get("reviews_without_meeting_id", 0),
        "eligible_examples": len(examples),
        "unique_meetings_used": len({x["meeting_id"] for x in examples}),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
        "dataset_fingerprint": dataset_fingerprint(examples) if examples else "",
        "split_mode": split_mode,
        "train_path": str(TRAIN_PATH),
        "val_path": str(VAL_PATH),
        "test_path": str(TEST_PATH),
        "filters": {
            "min_rating": MIN_RATING,
            "require_approved": REQUIRE_APPROVED,
            "require_edited_summary": REQUIRE_EDITED_SUMMARY,
            "min_summary_chars": MIN_SUMMARY_CHARS,
        },
        "errors": all_errors,
    }

    STATS_PATH.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[INFO] Retraining dataset generation finished.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

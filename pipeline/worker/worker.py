import logging
import os
import time
from typing import Any

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline-worker")

DATA_API_URL = os.environ.get(
    "DATA_API_URL",
    "http://data-api.platform.svc.cluster.local:8000",
).rstrip("/")
ASR_URL = os.environ.get(
    "ASR_URL",
    "http://serving-asr-mlflow.platform.svc.cluster.local:8000",
).rstrip("/")
SUMMARIZER_URL = os.environ.get(
    "SUMMARIZER_URL",
    "http://serving-baseline-mlflow.platform.svc.cluster.local:8000",
).rstrip("/")

SOURCE_PREFIX = os.environ.get("SOURCE_PREFIX", "jitsi_recording:")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MEETING_LIMIT = int(os.environ.get("MEETING_LIMIT", "25"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "1800"))
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "60"))
RUN_ONCE = os.environ.get("RUN_ONCE", "true").lower() in {"1", "true", "yes"}
RETRY_FAILED_ASR = os.environ.get("RETRY_FAILED_ASR", "false").lower() in {"1", "true", "yes"}


def _request(method: str, url: str, **kwargs: Any) -> requests.Response:
    resp = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    resp.raise_for_status()
    return resp


def _list_meetings() -> list[dict[str, Any]]:
    resp = _request(
        "GET",
        f"{DATA_API_URL}/meetings",
        params={"limit": MEETING_LIMIT, "source_prefix": SOURCE_PREFIX},
    )
    return resp.json()


def _process_asr(meeting_id: str) -> None:
    log.info("Running ASR for meeting %s", meeting_id)
    _request(
        "POST",
        f"{ASR_URL}/process-meeting",
        json={"meeting_id": meeting_id, "language": "en"},
    )


def _process_summary(meeting_id: str) -> None:
    log.info("Running summarization for meeting %s", meeting_id)
    _request(
        "POST",
        f"{SUMMARIZER_URL}/process-meeting",
        json={"meeting_id": meeting_id},
    )


def _mark_completed(meeting_id: str) -> None:
    _request(
        "PATCH",
        f"{DATA_API_URL}/meetings/{meeting_id}/asr",
        json={"asr_status": "completed", "status": "completed"},
    )


def _should_run_asr(meeting: dict[str, Any]) -> bool:
    asr_status = meeting.get("asr_status")
    if meeting.get("transcript_id") and asr_status == "completed":
        return False
    if asr_status == "failed" and not RETRY_FAILED_ASR:
        return False
    return asr_status in {"not_requested", "queued", "failed", None}


def _should_run_summary(meeting: dict[str, Any]) -> bool:
    return bool(meeting.get("transcript_id")) and not meeting.get("summary_id")


def run_once() -> int:
    meetings = _list_meetings()
    processed_meetings = 0
    log.info("Scanned %d meeting(s)", len(meetings))

    for meeting in meetings:
        if processed_meetings >= BATCH_SIZE:
            break

        meeting_id = meeting["meeting_id"]
        did_work = False
        try:
            if _should_run_asr(meeting):
                _process_asr(meeting_id)
                did_work = True

                # Refresh the row so summarization only runs after the transcript exists.
                refreshed = _list_meetings()
                meeting = next((m for m in refreshed if m["meeting_id"] == meeting_id), meeting)

            if _should_run_summary(meeting):
                _process_summary(meeting_id)
                _mark_completed(meeting_id)
                did_work = True

            if did_work:
                processed_meetings += 1

        except Exception:
            log.exception("Failed to process meeting %s", meeting_id)

    log.info("Processed %d meeting(s)", processed_meetings)
    return processed_meetings


def main() -> None:
    while True:
        run_once()
        if RUN_ONCE:
            break
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

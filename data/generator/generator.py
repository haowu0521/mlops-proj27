import requests
import random
from datetime import datetime, timedelta, UTC

BASE_URL = "http://127.0.0.1:8000"

def run_once():
    now = datetime.now(UTC)

    meeting = requests.post(
        f"{BASE_URL}/meetings",
        json={
            "source": "synthetic_generator",
            "started_at": now.isoformat(),
            "ended_at": (now + timedelta(minutes=30)).isoformat(),
            "audio_object_key": "raw/audio/sample.wav",
            "status": "completed",
        },
        timeout=30,
    )
    meeting.raise_for_status()
    meeting_id = meeting.json()["meeting_id"]

    requests.post(
        f"{BASE_URL}/transcripts",
        json={
            "meeting_id": meeting_id,
            "transcript_text": "Alice: Ship by Friday. Bob: test cleanup. Carol: verify edits.",
            "transcript_object_key": f"processed/transcripts/{meeting_id}.json",
        },
        timeout=30,
    ).raise_for_status()

    requests.post(
        f"{BASE_URL}/summaries",
        json={
            "meeting_id": meeting_id,
            "model_version": "baseline-v1",
            "summary_text": "Team aligned on Friday ship and assigned tasks.",
            "action_item_text": "Bob tests cleanup; Carol verifies edits.",
        },
        timeout=30,
    ).raise_for_status()

    requests.post(
        f"{BASE_URL}/reviews",
        json={
            "meeting_id": meeting_id,
            "rating": random.randint(3, 5),
            "edited_summary": "Team agreed on Friday delivery and task ownership.",
            "edited_action_items": "Test cleanup and verify edits.",
        },
        timeout=30,
    ).raise_for_status()

    print(f"created meeting {meeting_id}")

if __name__ == "__main__":
    for _ in range(5):
        run_once()

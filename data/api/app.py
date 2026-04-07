import uuid
from datetime import datetime
from typing import Optional

import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

DB_CONFIG = {
    "host": "postgres",
    "port": 5432,
    "dbname": "jitsi_mlops",
    "user": "user",
    "password": "jitsi_postgres",
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

class MeetingCreate(BaseModel):
    source: str
    started_at: datetime
    ended_at: datetime
    audio_object_key: str
    status: str

class TranscriptCreate(BaseModel):
    meeting_id: str
    transcript_text: str
    transcript_object_key: str

class SummaryCreate(BaseModel):
    meeting_id: str
    model_version: str
    summary_text: str
    action_item_text: str

class ReviewCreate(BaseModel):
    meeting_id: str
    rating: int
    edited_summary: Optional[str] = None
    edited_action_items: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/meetings")
def create_meeting(payload: MeetingCreate):
    meeting_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO meetings (
            meeting_id, source, started_at, ended_at, audio_object_key, status
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            meeting_id,
            payload.source,
            payload.started_at,
            payload.ended_at,
            payload.audio_object_key,
            payload.status,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"meeting_id": meeting_id}

@app.post("/transcripts")
def create_transcript(payload: TranscriptCreate):
    transcript_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transcripts (
            transcript_id, meeting_id, transcript_text, transcript_object_key
        ) VALUES (%s, %s, %s, %s)
        """,
        (
            transcript_id,
            payload.meeting_id,
            payload.transcript_text,
            payload.transcript_object_key,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"transcript_id": transcript_id}

@app.post("/summaries")
def create_summary(payload: SummaryCreate):
    summary_id = str(uuid.uuid4())
    action_item_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO summaries (
            summary_id, meeting_id, model_version, summary_text
        ) VALUES (%s, %s, %s, %s)
        """,
        (
            summary_id,
            payload.meeting_id,
            payload.model_version,
            payload.summary_text,
        ),
    )
    cur.execute(
        """
        INSERT INTO action_items (
            action_item_id, meeting_id, item_text
        ) VALUES (%s, %s, %s)
        """,
        (
            action_item_id,
            payload.meeting_id,
            payload.action_item_text,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"summary_id": summary_id, "action_item_id": action_item_id}

@app.post("/reviews")
def create_review(payload: ReviewCreate):
    review_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reviews (
            review_id, meeting_id, rating, edited_summary, edited_action_items
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        (
            review_id,
            payload.meeting_id,
            payload.rating,
            payload.edited_summary,
            payload.edited_action_items,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"review_id": review_id}

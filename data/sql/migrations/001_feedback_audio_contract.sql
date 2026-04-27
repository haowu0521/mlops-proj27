BEGIN;

ALTER TABLE meetings
    ADD COLUMN IF NOT EXISTS audio_checksum TEXT,
    ADD COLUMN IF NOT EXISTS audio_duration_seconds INT,
    ADD COLUMN IF NOT EXISTS asr_status TEXT,
    ADD COLUMN IF NOT EXISTS asr_job_id TEXT,
    ADD COLUMN IF NOT EXISTS asr_last_error TEXT,
    ADD COLUMN IF NOT EXISTS asr_requested_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS asr_completed_at TIMESTAMPTZ;

UPDATE meetings
SET status = COALESCE(NULLIF(status, ''), 'scheduled');

UPDATE meetings
SET asr_status = CASE
    WHEN EXISTS (SELECT 1 FROM transcripts t WHERE t.meeting_id = meetings.meeting_id) THEN 'completed'
    ELSE 'not_requested'
END
WHERE asr_status IS NULL;

ALTER TABLE meetings
    ALTER COLUMN status SET DEFAULT 'scheduled',
    ALTER COLUMN status SET NOT NULL,
    ALTER COLUMN asr_status SET DEFAULT 'not_requested',
    ALTER COLUMN asr_status SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'meetings_status_check'
    ) THEN
        ALTER TABLE meetings ADD CONSTRAINT meetings_status_check
        CHECK (status IN ('scheduled', 'in_progress', 'completed', 'failed', 'canceled'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'meetings_asr_status_check'
    ) THEN
        ALTER TABLE meetings ADD CONSTRAINT meetings_asr_status_check
        CHECK (asr_status IN ('not_requested', 'queued', 'processing', 'completed', 'failed'));
    END IF;
END$$;

ALTER TABLE reviews
    ADD COLUMN IF NOT EXISTS reviewer_id TEXT,
    ADD COLUMN IF NOT EXISTS approved BOOLEAN,
    ADD COLUMN IF NOT EXISTS correction_label TEXT,
    ADD COLUMN IF NOT EXISTS review_notes TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

DELETE FROM transcripts WHERE meeting_id IS NULL;
DELETE FROM summaries WHERE meeting_id IS NULL;
DELETE FROM action_items WHERE meeting_id IS NULL;
DELETE FROM reviews WHERE meeting_id IS NULL;

UPDATE reviews r
SET edited_summary = COALESCE(
    NULLIF(r.edited_summary, ''),
    (
        SELECT s.summary_text
        FROM summaries s
        WHERE s.meeting_id = r.meeting_id
        ORDER BY s.created_at DESC
        LIMIT 1
    ),
    '[missing summary]'
);

UPDATE reviews r
SET edited_action_items = COALESCE(
    NULLIF(r.edited_action_items, ''),
    (
        SELECT ai.item_text
        FROM action_items ai
        WHERE ai.meeting_id = r.meeting_id
        ORDER BY ai.created_at DESC
        LIMIT 1
    ),
    '[missing action items]'
);

UPDATE reviews
SET reviewer_id = COALESCE(NULLIF(reviewer_id, ''), 'streamlit-dashboard'),
    approved = COALESCE(approved, FALSE),
    correction_label = COALESCE(NULLIF(correction_label, ''), 'none'),
    rating = COALESCE(rating, 3),
    updated_at = COALESCE(updated_at, created_at, NOW());

UPDATE reviews
SET correction_label = 'none'
WHERE correction_label NOT IN ('none', 'minor', 'major', 'rewrite');

UPDATE reviews
SET rating = 3
WHERE rating < 1 OR rating > 5;

ALTER TABLE reviews
    ALTER COLUMN reviewer_id SET DEFAULT 'streamlit-dashboard',
    ALTER COLUMN correction_label SET DEFAULT 'none',
    ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE transcripts
    ALTER COLUMN meeting_id SET NOT NULL;

ALTER TABLE summaries
    ALTER COLUMN meeting_id SET NOT NULL;

ALTER TABLE action_items
    ALTER COLUMN meeting_id SET NOT NULL;

ALTER TABLE reviews
    ALTER COLUMN meeting_id SET NOT NULL,
    ALTER COLUMN reviewer_id SET NOT NULL,
    ALTER COLUMN approved SET NOT NULL,
    ALTER COLUMN correction_label SET NOT NULL,
    ALTER COLUMN edited_summary SET NOT NULL,
    ALTER COLUMN edited_action_items SET NOT NULL,
    ALTER COLUMN rating SET NOT NULL,
    ALTER COLUMN updated_at SET NOT NULL;

ALTER TABLE reviews DROP CONSTRAINT IF EXISTS reviews_correction_label_check;
ALTER TABLE reviews
    ADD CONSTRAINT reviews_correction_label_check CHECK (correction_label IN ('none', 'minor', 'major', 'rewrite'));

ALTER TABLE reviews DROP CONSTRAINT IF EXISTS reviews_rating_check;
ALTER TABLE reviews
    ADD CONSTRAINT reviews_rating_check CHECK (rating BETWEEN 1 AND 5);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'reviews_meeting_id_key'
    ) THEN
        ALTER TABLE reviews ADD CONSTRAINT reviews_meeting_id_key UNIQUE (meeting_id);
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_reviews_meeting_id ON reviews (meeting_id);
CREATE INDEX IF NOT EXISTS idx_reviews_approved_created ON reviews (approved, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcripts_meeting_created ON transcripts (meeting_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting_created ON summaries (meeting_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_items_meeting_created ON action_items (meeting_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meetings_asr_status ON meetings (asr_status, created_at ASC);

COMMIT;

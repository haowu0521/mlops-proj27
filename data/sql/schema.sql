CREATE TABLE IF NOT EXISTS meetings (
    meeting_id UUID PRIMARY KEY,
    source TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    audio_object_key TEXT,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    transcript_text TEXT,
    transcript_object_key TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    model_version TEXT,
    summary_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS action_items (
    action_item_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    item_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    rating INT,
    edited_summary TEXT,
    edited_action_items TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version TEXT PRIMARY KEY,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

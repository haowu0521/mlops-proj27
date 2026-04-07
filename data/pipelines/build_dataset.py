import json
import os
from datetime import datetime, UTC

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "jitsi_mlops",
    "user": "user",
    "password": "jitsi_postgres",
}

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

def main():
    dataset_version = datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = f"output/{dataset_version}"
    os.makedirs(out_dir, exist_ok=True)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            m.meeting_id,
            t.transcript_text,
            s.summary_text,
            r.edited_summary,
            r.edited_action_items,
            r.rating
        FROM meetings m
        JOIN transcripts t ON m.meeting_id = t.meeting_id
        JOIN summaries s ON m.meeting_id = s.meeting_id
        JOIN reviews r ON m.meeting_id = r.meeting_id
        ORDER BY m.created_at
        """
    )
    rows = cur.fetchall()

    examples = []
    for meeting_id, transcript_text, model_summary, edited_summary, edited_action_items, rating in rows:
        examples.append({
            "dataset_version": dataset_version,
            "meeting_id": str(meeting_id),
            "input_transcript": transcript_text,
            "target_summary": edited_summary or model_summary,
            "target_action_items": edited_action_items,
            "model_summary": model_summary,
            "rating": rating,
            "source": "production_feedback",
        })

    n = len(examples)
    train_end = max(1, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85)) if n > 2 else n

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    write_jsonl(f"{out_dir}/train.jsonl", train)
    write_jsonl(f"{out_dir}/val.jsonl", val)
    write_jsonl(f"{out_dir}/test.jsonl", test)

    with open(f"{out_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_version": dataset_version,
            "created_at": datetime.now(UTC).isoformat(),
            "num_examples": n,
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
        }, f, indent=2)

    cur.execute(
        "INSERT INTO dataset_versions (dataset_version, source) VALUES (%s, %s)",
        (dataset_version, "production_feedback"),
    )
    conn.commit()
    cur.close()
    conn.close()

    print(dataset_version)

if __name__ == "__main__":
    main()

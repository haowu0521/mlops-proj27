import json
import os
import subprocess
from datetime import datetime, UTC
from glob import glob


RAW_DIR = "external_data/qmsum_raw"
OUTPUT_BASE = "output_external"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_transcript(meeting_transcripts):
    parts = []
    for turn in meeting_transcripts:
        speaker = turn.get("speaker", "Speaker")
        content = turn.get("content", "")
        parts.append(f"{speaker}: {content}".strip())
    return "\n".join(parts).strip()


def normalize_one_doc(doc, dataset_version):
    meeting_id = doc.get("meeting_id", "unknown_meeting")
    transcript = flatten_transcript(doc.get("meeting_transcripts", []))
    rows = []

    for idx, q in enumerate(doc.get("general_query_list", [])):
        query_text = q.get("query", "general_summary")
        target_summary = q.get("answer", "")

        if not transcript or not target_summary:
            continue

        rows.append(
            {
                "dataset_version": dataset_version,
                "source": "QMSum",
                "source_meeting_id": meeting_id,
                "query_id": f"{meeting_id}_general_{idx}",
                "query_text": query_text,
                "input_transcript": transcript,
                "target_summary": target_summary,
                "target_action_items": [],
                "split": "train",
            }
        )

    return rows


def main():
    dataset_version = datetime.now(UTC).strftime("qmsum_v%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, dataset_version)
    os.makedirs(output_dir, exist_ok=True)

    raw_files = sorted(glob(os.path.join(RAW_DIR, "*.json")))
    if not raw_files:
        raise FileNotFoundError(f"No JSON files found in {RAW_DIR}")

    normalized = []
    docs_seen = 0

    for path in raw_files:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        docs_seen += 1
        normalized.extend(normalize_one_doc(doc, dataset_version))

    jsonl_path = os.path.join(output_dir, "qmsum_train.jsonl")
    write_jsonl(jsonl_path, normalized)

    manifest = {
        "dataset_version": dataset_version,
        "source_dataset": "QMSum",
        "raw_dir": RAW_DIR,
        "raw_files_seen": docs_seen,
        "records_after_normalization": len(normalized),
        "output_file": jsonl_path,
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_selection": {
            "query_type": "general_query_list",
            "require_nonempty_transcript": True,
            "require_nonempty_summary": True,
        },
    }

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    subprocess.run(
        [
            "aws",
            "--endpoint-url", "http://127.0.0.1:9000",
            "s3", "cp", "--recursive",
            output_dir,
            f"s3://jitsi-data/external/qmsum/{dataset_version}/"
        ],
        check=True
    )

    print(f"Built QMSum dataset version: {dataset_version}")
    print(f"Rows written: {len(normalized)}")
    print(f"Uploaded to: s3://jitsi-data/external/qmsum/{dataset_version}/")


if __name__ == "__main__":
    main()

import json
import argparse
from pathlib import Path
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def asr_output_to_summarization_input(asr_output: Dict[str, Any]) -> Dict[str, Any]:
    if "meeting_id" not in asr_output:
        raise ValueError("ASR output is missing 'meeting_id'")
    if "transcript" not in asr_output:
        raise ValueError("ASR output is missing 'transcript'")

    transcript = str(asr_output["transcript"]).strip()
    if not transcript:
        raise ValueError("ASR output contains an empty transcript")

    summarization_input = {
        "meeting_id": asr_output["meeting_id"],
        "transcript": transcript
    }

    return summarization_input


def main():
    parser = argparse.ArgumentParser(
        description="Convert ASR output JSON to summarization input JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ASR output JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to summarization input JSON file",
    )
    args = parser.parse_args()

    asr_output = load_json(args.input)
    summarization_input = asr_output_to_summarization_input(asr_output)
    save_json(summarization_input, args.output)

    print("Conversion completed successfully.")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

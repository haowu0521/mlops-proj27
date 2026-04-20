import os
import tempfile
from pathlib import Path

import boto3
import mlflow.pyfunc
import pandas as pd


def main():
    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "http://129.114.26.182:30900")
    minio_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    minio_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

    bucket_name = os.environ.get("MINIO_BUCKET", "audio-files")
    object_key = os.environ.get("MINIO_OBJECT_KEY", "")
    meeting_id = os.environ.get("MEETING_ID", "demo_001")
    language = os.environ.get("ASR_LANGUAGE", "en")

    model_uri = os.environ.get("ASR_MODEL_URI", "models:/jitsi-asr/1")

    if not object_key:
        raise ValueError(
            "MINIO_OBJECT_KEY is required. Example: export MINIO_OBJECT_KEY=test_audio.wav"
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    suffix = Path(object_key).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        local_audio_path = tmp.name

    try:
        print(f"Downloading s3://{bucket_name}/{object_key} -> {local_audio_path}")
        s3.download_file(bucket_name, object_key, local_audio_path)

        print(f"Loading ASR model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        df = pd.DataFrame(
            [
                {
                    "meeting_id": meeting_id,
                    "audio_path": local_audio_path,
                    "language": language,
                    "source": "minio_audio",
                }
            ]
        )

        print("Running ASR inference...")
        result = model.predict(df)

        print("\nRaw DataFrame result:")
        print(result)

        print("\nResult as dict records:")
        print(result.to_dict(orient="records"))

    finally:
        try:
            os.remove(local_audio_path)
            print(f"\nTemporary file removed: {local_audio_path}")
        except OSError:
            pass


if __name__ == "__main__":
    main()

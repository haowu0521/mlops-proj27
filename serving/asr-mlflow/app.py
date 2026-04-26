# ---------------------------------------------------------------------------
# asr-mlflow — FastAPI ASR service that loads the jitsi-asr@production model
# from the MLflow Model Registry via the pyfunc flavor.
#
# Mirrors serving/baseline-mlflow/app.py for symmetry. Differences:
#   - The pyfunc model under the hood is a faster-whisper WhisperModel, which
#     reads audio from a *local file path*, not raw bytes. So /predict accepts
#     a MinIO object key, downloads the file to /tmp, and feeds the path to
#     the model. The temp file is deleted in `finally` so we don't leak.
#   - Input schema: {meeting_id, audio_object_key, audio_bucket?, language?}
#     Output schema: {meeting_id, language, transcript, segments[]}
#   - faster-whisper itself downloads the whisper-small weights from
#     HuggingFace Hub on first call (cached at ~/.cache/huggingface). Cold
#     start is ~30-90s on CPU; readiness probe is set tolerant accordingly.
#
# Promoting a new ASR model:
#   bump the @production alias in MLflow, then
#     sudo kubectl rollout restart deployment/serving-asr-mlflow -n platform
#
# Env vars:
#   MLFLOW_TRACKING_URI     - http://mlflow.platform.svc.cluster.local:5000
#   MLFLOW_S3_ENDPOINT_URL  - http://minio.platform.svc.cluster.local:9000
#   AWS_ACCESS_KEY_ID       - MinIO access key (also used for audio download)
#   AWS_SECRET_ACCESS_KEY   - MinIO secret key
#   MODEL_NAME              - registered model name (default: jitsi-asr)
#   MODEL_ALIAS             - registry alias (default: production)
#   AUDIO_BUCKET            - default MinIO bucket for /predict (default: recordings)
#   TRANSCRIPT_BUCKET       - bucket for ASR JSON artifacts (default: jitsi-data)
#   DATA_API_URL            - internal data-api URL for /process-* integration
# ---------------------------------------------------------------------------

import glob
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import boto3
import mlflow.pyfunc
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving-asr")

MODEL_NAME = os.environ.get("MODEL_NAME", "jitsi-asr")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# S3/MinIO config — reuse the same env vars MLflow already needs.
S3_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
DEFAULT_BUCKET = os.environ.get("AUDIO_BUCKET", "recordings")
TRANSCRIPT_BUCKET = os.environ.get("TRANSCRIPT_BUCKET", "jitsi-data")
TRANSCRIPT_PREFIX = os.environ.get("TRANSCRIPT_PREFIX", "transcripts").strip("/")
DATA_API_URL = os.environ.get(
    "DATA_API_URL",
    "http://data-api.platform.svc.cluster.local:8000",
).rstrip("/")
DATA_API_TIMEOUT_SECONDS = float(os.environ.get("DATA_API_TIMEOUT_SECONDS", "30"))

app = FastAPI(title="jitsi-asr (MLflow-backed faster-whisper)")


def _load_pyfunc_model():
    """Resolve the alias -> version for logging, then load via pyfunc.

    The MlflowClient call is purely for observability — if it fails we
    still try the pyfunc load, which is the thing that actually matters.
    """
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        log.info("Alias %s -> version %s (run_id=%s)", MODEL_ALIAS, mv.version, mv.run_id)
    except Exception as e:
        log.warning("Could not resolve alias metadata (non-fatal): %s", e)

    log.info("Loading pyfunc model: %s", MODEL_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    log.info("Model loaded; ready to serve.")
    return model


# Load once at startup. If this fails, the pod crashes and k8s reschedules —
# which is what we want for a broken model promotion.
_model = _load_pyfunc_model()

# boto3 picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY automatically.
_s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT)

# Belt-and-suspenders: clean up any stale temp files left over from a
# previous pod that was SIGKILLed mid-transcribe (so /tmp's `finally`
# block never had a chance to run). Combined with the sizeLimit set on
# the /tmp emptyDir in the Deployment manifest, this means a runaway
# leak can never escape the pod and pollute /mnt/block on the host.
def _cleanup_stale_tempfiles():
    pattern = os.path.join(tempfile.gettempdir(), "asr-input-*")
    for stale in glob.glob(pattern):
        try:
            os.remove(stale)
            log.info("Cleaned up stale temp file: %s", stale)
        except Exception as e:
            log.warning("Could not remove stale temp file %s: %s", stale, e)


_cleanup_stale_tempfiles()


class PredictRequest(BaseModel):
    meeting_id: str
    audio_object_key: str
    audio_bucket: str = DEFAULT_BUCKET
    language: Optional[str] = None  # None = auto-detect


class Segment(BaseModel):
    start: float
    end: float
    text: str


class PredictResponse(BaseModel):
    meeting_id: str
    language: str
    transcript: str
    segments: List[Segment]
    model_name: str = MODEL_NAME
    model_alias: str = MODEL_ALIAS


class ProcessMeetingRequest(BaseModel):
    meeting_id: str
    audio_object_key: Optional[str] = None
    audio_bucket: str = DEFAULT_BUCKET
    language: Optional[str] = None


class ProcessMeetingResponse(PredictResponse):
    transcript_id: str
    transcript_object_key: str


def _download_audio(bucket: str, key: str) -> Path:
    """Download audio from S3/MinIO to a temp file. Returns the local path."""
    suffix = Path(key).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(prefix="asr-input-", suffix=suffix, delete=False)
    tmp.close()
    log.info("Downloading s3://%s/%s -> %s", bucket, key, tmp.name)
    _s3.download_file(bucket, key, tmp.name)
    return Path(tmp.name)


def _run_asr(req: PredictRequest) -> PredictResponse:
    audio_path = None
    try:
        audio_path = _download_audio(req.audio_bucket, req.audio_object_key)

        # Build the DataFrame the pyfunc wrapper expects. `language=None`
        # becomes NaN in pandas, and the wrapper checks pd.notna() before
        # using it — so None means "auto-detect".
        df = pd.DataFrame([{
            "meeting_id": req.meeting_id,
            "audio_path": str(audio_path),
            "language": req.language,
        }])

        result_df = _model.predict(df)
        row = result_df.iloc[0]

        segments_raw = row["segments"] or []
        return PredictResponse(
            meeting_id=req.meeting_id,
            language=str(row["language"]),
            transcript=str(row["transcript"]),
            segments=[Segment(**s) for s in segments_raw],
        )
    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception as cleanup_err:
                log.warning("Failed to clean up %s: %s", audio_path, cleanup_err)


def _data_api(method: str, path: str, **kwargs: Any) -> requests.Response:
    resp = requests.request(
        method,
        f"{DATA_API_URL}{path}",
        timeout=DATA_API_TIMEOUT_SECONDS,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def _set_asr_status(
    meeting_id: str,
    asr_status: str,
    *,
    status: Optional[str] = None,
    asr_job_id: Optional[str] = None,
    asr_last_error: Optional[str] = None,
) -> None:
    payload = {
        "asr_status": asr_status,
        "status": status,
        "asr_job_id": asr_job_id,
        "asr_last_error": asr_last_error,
    }
    _data_api("PATCH", f"/meetings/{meeting_id}/asr", json={k: v for k, v in payload.items() if v is not None})


def _get_audio_object_key(meeting_id: str) -> str:
    resp = _data_api("GET", f"/meetings/{meeting_id}/audio")
    audio_object_key = resp.json().get("audio_object_key")
    if not audio_object_key:
        raise RuntimeError(f"meeting {meeting_id} has no audio_object_key")
    return audio_object_key


def _store_transcript_artifact(result: PredictResponse) -> str:
    key = f"{TRANSCRIPT_PREFIX}/{result.meeting_id}.json" if TRANSCRIPT_PREFIX else f"{result.meeting_id}.json"
    body = {
        "meeting_id": result.meeting_id,
        "language": result.language,
        "transcript": result.transcript,
        "segments": [segment.model_dump() for segment in result.segments],
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }
    _s3.put_object(
        Bucket=TRANSCRIPT_BUCKET,
        Key=key,
        Body=json.dumps(body, ensure_ascii=True).encode("utf-8"),
        ContentType="application/json",
    )
    return key


def _create_transcript(meeting_id: str, transcript_text: str, transcript_object_key: str) -> str:
    resp = _data_api(
        "POST",
        "/transcripts",
        json={
            "meeting_id": meeting_id,
            "transcript_text": transcript_text,
            "transcript_object_key": transcript_object_key,
        },
    )
    return str(resp.json()["transcript_id"])


def _process_meeting(req: ProcessMeetingRequest) -> ProcessMeetingResponse:
    audio_object_key = req.audio_object_key or _get_audio_object_key(req.meeting_id)
    try:
        _set_asr_status(
            req.meeting_id,
            "processing",
            status="in_progress",
        )
        result = _run_asr(PredictRequest(
            meeting_id=req.meeting_id,
            audio_object_key=audio_object_key,
            audio_bucket=req.audio_bucket,
            language=req.language,
        ))
        transcript_object_key = _store_transcript_artifact(result)
        transcript_id = _create_transcript(
            req.meeting_id,
            result.transcript,
            transcript_object_key,
        )
        _set_asr_status(req.meeting_id, "completed")
        return ProcessMeetingResponse(
            **result.model_dump(),
            transcript_id=transcript_id,
            transcript_object_key=transcript_object_key,
        )
    except Exception as e:
        log.exception("process meeting failed")
        try:
            _set_asr_status(
                req.meeting_id,
                "failed",
                status="failed",
                asr_last_error=str(e)[:500],
            )
        except Exception as status_err:
            log.warning("Failed to record ASR failure for meeting %s: %s", req.meeting_id, status_err)
        raise


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Download audio from MinIO, run faster-whisper, return transcript."""
    try:
        return _run_asr(req)
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-meeting", response_model=ProcessMeetingResponse)
def process_meeting(req: ProcessMeetingRequest):
    """Run ASR for one meeting and persist the transcript through data-api."""
    try:
        return _process_meeting(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

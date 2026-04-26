# ---------------------------------------------------------------------------
# baseline-mlflow — FastAPI summarizer that loads the jitsi-summarizer model
# from the MLflow Model Registry via the pyfunc flavor.
#
# Why pyfunc (again):
#   Earlier versions of this file bypassed pyfunc and loaded the raw HF
#   artifact directly, because v1 of the registered model was pickled with
#   pipeline("summarization") — a task removed from transformers 5.x's
#   pipeline registry, so load_context() would blow up at startup.
#   Training fixed the bug (pipeline("text2text-generation") in train.py)
#   and registered v7; production alias now points at v7, so the pyfunc
#   path actually loads again. Using it here keeps serving "MLflow-native":
#   promoting a new alias in the registry is the only step needed to ship
#   a new model.
#
# Generation params are no longer controlled here:
#   SummarizationPyFuncModel.predict() hardcodes max_new_tokens=128 and
#   uses pipeline defaults for beams / no_repeat_ngram_size. If summaries
#   come out too short or too repetitive, fix it in train.py and retrain —
#   not here. (Env vars MAX_NEW_TOKENS / NUM_BEAMS from the old workaround
#   are intentionally removed so nobody thinks they still do anything.)
#
# Env vars:
#   MLFLOW_TRACKING_URI     - http://mlflow.platform.svc.cluster.local:5000
#   MLFLOW_S3_ENDPOINT_URL  - http://minio.platform.svc.cluster.local:9000
#   AWS_ACCESS_KEY_ID       - MinIO access key
#   AWS_SECRET_ACCESS_KEY   - MinIO secret key
#   MODEL_NAME              - registered model name (default: jitsi-summarizer)
#   MODEL_ALIAS             - registry alias (default: production)
#   DATA_API_URL            - internal data-api URL for /process-meeting
# ---------------------------------------------------------------------------

import logging
import os
import re
from typing import Any, List, Tuple

import mlflow.pyfunc
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

MODEL_NAME = os.environ.get("MODEL_NAME", "jitsi-summarizer")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
DATA_API_URL = os.environ.get(
    "DATA_API_URL",
    "http://data-api.platform.svc.cluster.local:8000",
).rstrip("/")
DATA_API_TIMEOUT_SECONDS = float(os.environ.get("DATA_API_TIMEOUT_SECONDS", "30"))

app = FastAPI(title="jitsi-summarizer (MLflow-backed)")


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


class PredictRequest(BaseModel):
    meeting_id: str
    transcript: str


class PredictResponse(BaseModel):
    meeting_id: str
    summary: str
    action_items: List[str]
    model_name: str = MODEL_NAME
    model_alias: str = MODEL_ALIAS


class ProcessMeetingRequest(BaseModel):
    meeting_id: str


class ProcessMeetingResponse(PredictResponse):
    summary_id: str
    action_item_id: str


def _generate(text: str) -> str:
    """Call the pyfunc. Training-side predict() expects a DataFrame with a
    'text' column and returns a DataFrame with a 'summary' column
    (see training/train.py:SummarizationPyFuncModel.predict)."""
    result_df = _model.predict(pd.DataFrame({"text": [text]}))
    return str(result_df["summary"].iloc[0])


def _split_action_items(generated: str) -> Tuple[str, List[str]]:
    """Preserve the same output contract as serving/baseline/app.py:
    split the generated text on the 'Action Items:' marker (case-insensitive)
    into (summary, [action items])."""
    match = re.search(r"action items:", generated, flags=re.IGNORECASE)
    if not match:
        return generated.strip(), []
    summary = generated[: match.start()].strip()
    tail = generated[match.end():]
    items = [s.strip() for s in tail.split(".") if s.strip()]
    return summary, items


def _data_api(method: str, path: str, **kwargs: Any) -> requests.Response:
    resp = requests.request(
        method,
        f"{DATA_API_URL}{path}",
        timeout=DATA_API_TIMEOUT_SECONDS,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def _summarize(meeting_id: str, transcript: str) -> PredictResponse:
    generated = _generate(transcript)
    summary, action_items = _split_action_items(generated)
    return PredictResponse(
        meeting_id=meeting_id,
        summary=summary,
        action_items=action_items,
    )


def _process_meeting(meeting_id: str) -> ProcessMeetingResponse:
    transcript_resp = _data_api("GET", f"/transcripts/by_meeting/{meeting_id}")
    transcript = transcript_resp.json().get("transcript_text")
    if not transcript:
        raise RuntimeError(f"meeting {meeting_id} has no transcript_text")

    prediction = _summarize(meeting_id, transcript)
    summary_resp = _data_api(
        "POST",
        "/summaries",
        json={
            "meeting_id": meeting_id,
            "model_version": f"{MODEL_NAME}@{MODEL_ALIAS}",
            "summary_text": prediction.summary,
            "action_item_text": "\n".join(prediction.action_items),
        },
    )
    ids = summary_resp.json()

    return ProcessMeetingResponse(
        **prediction.model_dump(),
        summary_id=str(ids["summary_id"]),
        action_item_id=str(ids["action_item_id"]),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _summarize(req.meeting_id, req.transcript)
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-meeting", response_model=ProcessMeetingResponse)
def process_meeting(req: ProcessMeetingRequest):
    """Fetch the latest transcript for a meeting, summarize it, and persist the result."""
    try:
        return _process_meeting(req.meeting_id)
    except Exception as e:
        log.exception("process meeting failed")
        raise HTTPException(status_code=500, detail=str(e))

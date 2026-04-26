# Python 3.12 to match the training env captured in the MLflow model's
# python_env.yaml (same reasoning as serving/baseline-mlflow).
FROM python:3.12-slim

WORKDIR /app

# ffmpeg is required by faster-whisper to decode mp4/mov/m4a/etc. into PCM.
# curl is kept for health-check style probing inside the container if needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

EXPOSE 8000

# The whisper-small weights (~500MB int8) are downloaded from HuggingFace
# Hub on first /predict call by faster-whisper. They are cached at
# ~/.cache/huggingface/hub. If you want to avoid the cold-start download,
# add `RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"`
# above to bake them into the image (image grows by ~500MB).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

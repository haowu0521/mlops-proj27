FROM python:3.10-slim
WORKDIR /app

# 1. Install system build dependencies required by ONNX/Optimum
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip to handle heavy ML wheels properly
RUN pip install --no-cache-dir --upgrade pip

# 3. Install FastAPI and the ONNX runtime dependencies
RUN pip install --no-cache-dir fastapi uvicorn optimum[onnxruntime] transformers pyyaml pydantic

# 4. Copy the app and the heavily optimized model folder
COPY app.py /app/
COPY onnx_model /app/onnx_model/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

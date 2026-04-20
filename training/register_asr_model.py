import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from faster_whisper import WhisperModel


class FasterWhisperPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_size_or_path = context.artifacts["asr_model"]
        compute_type = context.model_config.get("compute_type", "int8")
        device = context.model_config.get("device", "cpu")

        self.model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if "audio_path" not in model_input.columns:
            raise ValueError("Input DataFrame must contain an 'audio_path' column.")

        language = None
        if "language" in model_input.columns and len(model_input["language"]) > 0:
            language = model_input["language"].iloc[0]

        results = []
        for _, row in model_input.iterrows():
            audio_path = str(row["audio_path"])
            meeting_id = str(row["meeting_id"]) if "meeting_id" in row else None

            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
            )

            segment_list = []
            full_text = []
            for seg in segments:
                seg_text = seg.text.strip()
                full_text.append(seg_text)
                segment_list.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg_text,
                    }
                )

            results.append(
                {
                    "meeting_id": meeting_id,
                    "language": info.language,
                    "transcript": " ".join(full_text),
                    "segments": segment_list,
                }
            )

        return pd.DataFrame(results)


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.26.182:30500")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    # 你们共享平台如果还是 MinIO，需要这几个环境变量先 export 好
    # export MLFLOW_S3_ENDPOINT_URL=http://129.114.26.182:30900
    # export AWS_ACCESS_KEY_ID=minio
    # export AWS_SECRET_ACCESS_KEY=minio123

    experiment_name = "jitsi-asr"
    registered_model_name = "jitsi-asr"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    input_example = pd.DataFrame(
        {
            "meeting_id": ["demo_001"],
            "audio_path": ["/tmp/demo.wav"],
            "language": ["en"],
        }
    )

    with mlflow.start_run(run_name="faster_whisper_small_register"):
        model_name_or_path = "small"

        mlflow.log_param("asr_backend", "faster-whisper")
        mlflow.log_param("model_name_or_path", model_name_or_path)
        mlflow.log_param("device", "cpu")
        mlflow.log_param("compute_type", "int8")

        model_info = mlflow.pyfunc.log_model(
            artifact_path="registered_asr_model",
            python_model=FasterWhisperPyFuncModel(),
            artifacts={"asr_model": model_name_or_path},
            input_example=input_example,
            registered_model_name=registered_model_name,
            pip_requirements=[
                "mlflow==2.19.0",
                "faster-whisper",
                "pandas",
            ],
            model_config={
                "device": "cpu",
                "compute_type": "int8",
            },
        )

        print("ASR model registered successfully.")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Registered model name: {registered_model_name}")


if __name__ == "__main__":
    main()

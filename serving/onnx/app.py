import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

class MeetingInput(BaseModel):
    meeting_id: str
    transcript: str

class MeetingOutput(BaseModel):
    meeting_id: str
    summary: str
    action_items: list[str]

app = FastAPI()

# Load the local ONNX model we just exported
MODEL_PATH = "onnx_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

@app.post("/predict", response_model=MeetingOutput)
def predict_summary(request: MeetingInput):
    try:
        prompt = f"Summarize the following meeting and list the action items:\n\n{request.transcript}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        # Inference using ONNX Runtime
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Action Items:" in generated_text or "action items:" in generated_text.lower():
            parts = generated_text.lower().split("action items:")
            summary = parts[0].strip()
            action_items = [i.strip() for i in parts[1].split(".") if i.strip()]
        else:
            summary = generated_text.strip()
            action_items = []

        return MeetingOutput(meeting_id=request.meeting_id, summary=summary, action_items=action_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

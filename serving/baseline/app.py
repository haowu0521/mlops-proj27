import yaml
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

with open("serving_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["model_name_or_path"]

class MeetingInput(BaseModel):
    meeting_id: str
    transcript: str

class MeetingOutput(BaseModel):
    meeting_id: str
    summary: str
    action_items: list[str]

app = FastAPI()
device = torch.device("cpu") # Forcing CPU for the VM baseline

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

@app.post("/predict", response_model=MeetingOutput)
def predict_summary(request: MeetingInput):
    try:
        prompt = f"Summarize the following meeting and list the action items:\n\n{request.transcript}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Action Items:" in generated_text or "action items:" in generated_text.lower():
            parts = generated_text.lower().split("action items:")
            summary, action_items = parts[0].strip(), [i.strip() for i in parts[1].split(".") if i.strip()]
        else:
            summary, action_items = generated_text.strip(), []

        return MeetingOutput(meeting_id=request.meeting_id, summary=summary, action_items=action_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

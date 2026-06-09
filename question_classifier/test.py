from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

MODEL_PATH = Path("models/roberta-base/final")
ID2LABEL = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()

question = "오늘 날씨 어때?"

inputs = tokenizer(
    question,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

pred_id = torch.argmax(outputs.logits, dim=-1).item()
print(f"질문: {question}")
print(f"분류: {ID2LABEL[pred_id]}")
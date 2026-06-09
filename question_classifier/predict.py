from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

MODEL_PATH = Path("models/roberta-base/final")
ID2LABEL = {0: "non-menu", 1: "menu-only", 2: "menu-complex", 3: "invalid"}
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()

def predict(question: str) -> str:
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
    return ID2LABEL[pred_id]

questions = [
    "조용하고 콘센트 있는 카페 추천해줘",
    "아메리카노 맛있는 카페 어디야?",
    "아아 맛집 추천해줘",
    "라떼 마시면서 공부하기 좋은 카페 추천해줘",
    "사장님이 이쁜 카페 추천해줘",
    "강남 맛집 추천해줘",
    "오늘 날씨 어때?",
    "근처 식당 알려줘",
    "카페 말고 술집 추천해줘",
    "이 카페 몇시에 열어?",
    "북태평양 기단이 머무르는 기간은?",
    "이 카페 신고하고 싶은데 방법을 추천좀 해줘",
    "이 카페 콘센트가 어디 있는지 알려줘",
    "키공 하기 위해 콘센트 좀 있는 카페 알려줘",
    "아메리카노 맛집 알려줘"
]

for q in questions:
    print(f"[{predict(q)}] {q}")
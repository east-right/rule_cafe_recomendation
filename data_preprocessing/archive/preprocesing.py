import os
import json
import asyncio
import re
import random
import pandas as pd
import emoji
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from typing import Literal
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from prompt import system_prompt

load_dotenv()

# 설정
SAVE_DIR = "../data"
INPUT_FILE = "../data/shinline_cafe_reviews_test.csv"
OUTPUT_FILE = os.path.join(SAVE_DIR, "ner_result.json")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL = "gpt-4o-mini"
CONCURRENCY = 5
MAX_RETRIES = 6
BASE_WAIT = 1.0
CHECKPOINT_EVERY = 50

# 전처리
emoji_pattern = re.compile(
    '['
    '\U00010000-\U0010ffff'
    '\u2600-\u27BF'
    ']',
    flags=re.UNICODE
)

def review_cleaner(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace=" ")
    text = emoji_pattern.sub(" ", text)
    text = re.sub(r'([^\w\s^])\1+', r'\1', text)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df['리뷰내용'] = df['리뷰내용'].apply(review_cleaner)
    df = df[df['리뷰내용'].str.strip() != ""].reset_index(drop=True)
    print(f"데이터 로드 완료: {before}건 → 전처리 후 {len(df)}건")
    return df

# Pydantic 스키마
class EntityItem(BaseModel):
    entity: str
    descriptor: str | None
    sentiment: Literal["긍정", "부정", "중립"]

class NERResult(BaseModel):
    MENU: list[EntityItem]
    FACILITY: list[EntityItem]
    ATMOSPHERE: list[EntityItem]
    TARGET: list[EntityItem]

# 멀티턴 퓨샷 방법
FEW_SHOT: list[dict] = [
    {
        "role": "user",
        "content": "루프탑 뷰가 너무 좋고 크로플이 맛있어요. 데이트 코스로 강추!"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "MENU": [{"entity": "크로플", "descriptor": "맛있다", "sentiment": "긍정"}],
            "FACILITY": [{"entity": "루프탑", "descriptor": "뷰가 좋다", "sentiment": "긍정"}],
            "ATMOSPHERE": [],
            "TARGET": [{"entity": "데이트 코스", "descriptor": None, "sentiment": "긍정"}]
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "내부는 넓고 인테리어도 이쁜데 화장실은 청결하지 않고 벌레가 엄청나게 많아요."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "MENU": [],
            "FACILITY": [
                {"entity": "내부", "descriptor": "넓다", "sentiment": "긍정"},
                {"entity": "인테리어", "descriptor": "이쁘다", "sentiment": "긍정"},
                {"entity": "화장실", "descriptor": "청결하지 않다", "sentiment": "부정"},
            ],
            "ATMOSPHERE": [],
            "TARGET": []
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "쿠키 프라페가 달긴 하지만 맛있어요. 콘센트가 많아서 카공하기 좋고 사장님도 친절해요."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "MENU": [{"entity": "쿠키 프라페", "descriptor": "맛있다", "sentiment": "긍정"}],
            "FACILITY": [{"entity": "콘센트", "descriptor": "많다", "sentiment": "긍정"}],
            "ATMOSPHERE": [],
            "TARGET": [{"entity": "카공", "descriptor": "좋다", "sentiment": "긍정"}]
        }, ensure_ascii=False)
    },
]

# 체크포인트
def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 단건 추출 (비동기 + retry)
async def extract_ner(
    item: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.beta.chat.completions.parse(
                    model=MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *FEW_SHOT,
                        {"role": "user", "content": item["리뷰내용"]}
                    ],
                    response_format=NERResult,
                )
                ner: NERResult = response.choices[0].message.parsed
                return {
                    "index": item["index"],
                    "original_text": item["리뷰내용"],
                    "ner_tags": ner.model_dump(),
                }

            except RateLimitError as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"\n[Index {item.get('index')}] 재시도 초과, 건너뜀: {e}")
                    return None
                wait = BASE_WAIT * (2 ** attempt) + random.uniform(0, 1)
                print(f"\n[Index {item.get('index')}] Rate limit → {wait:.1f}초 후 재시도 ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)

            except Exception as e:
                print(f"\n[Index {item.get('index')}] 에러: {e}")
                return None

# 실행 부분
async def main():
    df = load_data(INPUT_FILE)
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(CONCURRENCY)

    results_dict = load_checkpoint(OUTPUT_FILE)
    print(f"기존 완료: {len(results_dict)}건")

    records = df.reset_index().to_dict("records")
    pending = [r for r in records if str(r["index"]) not in results_dict]
    print(f"처리 대상: {len(pending)}건 / 전체: {len(records)}건")

    if not pending:
        print("모두 완료된 상태입니다.")
        return

    tasks = [extract_ner(item, client, semaphore) for item in pending]

    completed = 0
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="NER 추출"):
        result = await coro
        if result is not None:
            results_dict[str(result["index"])] = result
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                save_checkpoint(results_dict, OUTPUT_FILE)

    save_checkpoint(results_dict, OUTPUT_FILE)
    print(f"\n완료. 총 {len(results_dict)}건 → {OUTPUT_FILE}")

asyncio.run(main())
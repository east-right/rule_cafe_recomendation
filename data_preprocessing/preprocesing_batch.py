import os
import json
import re
import time
import pandas as pd
import emoji
from openai import OpenAI
from dotenv import load_dotenv
from prompt import system_prompt

load_dotenv()

# ── 설정 ──────────────────────────────────────────
SAVE_DIR = "../data"
INPUT_FILE = "../data/shinline_cafe_reviews_test.csv"
OUTPUT_FILE = os.path.join(SAVE_DIR, "ner_result.json")
BATCH_INPUT_FILE = os.path.join(SAVE_DIR, "batch_input.jsonl")
BATCH_ID_FILE = os.path.join(SAVE_DIR, "batch_id.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 전처리 ─────────────────────────────────────────
emoji_pattern = re.compile(
    '[' '\U00010000-\U0010ffff' '\u2600-\u27BF' ']',
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

# ── Few-shot ───────────────────────────────────────
FEW_SHOT: list[dict] = [
    {"role": "user", "content": "루프탑 뷰가 너무 좋고 크로플이 맛있어요. 데이트 코스로 강추!"},
    {"role": "assistant", "content": json.dumps({
        "MENU": [{"entity": "크로플", "descriptor": "맛있다", "sentiment": "긍정"}],
        "FACILITY": [{"entity": "루프탑", "descriptor": "뷰가 좋다", "sentiment": "긍정"}],
        "ATMOSPHERE": [],
        "TARGET": [{"entity": "데이트 코스", "descriptor": None, "sentiment": "긍정"}]
    }, ensure_ascii=False)},
    {"role": "user", "content": "내부는 넓고 인테리어도 이쁜데 화장실은 청결하지 않고 벌레가 엄청나게 많아요."},
    {"role": "assistant", "content": json.dumps({
        "MENU": [],
        "FACILITY": [
            {"entity": "내부", "descriptor": "넓다", "sentiment": "긍정"},
            {"entity": "인테리어", "descriptor": "이쁘다", "sentiment": "긍정"},
            {"entity": "화장실", "descriptor": "청결하지 않다", "sentiment": "부정"},
        ],
        "ATMOSPHERE": [],
        "TARGET": []
    }, ensure_ascii=False)},
    {"role": "user", "content": "쿠키 프라페가 달긴 하지만 맛있어요. 콘센트가 많아서 카공하기 좋고 사장님도 친절해요."},
    {"role": "assistant", "content": json.dumps({
        "MENU": [{"entity": "쿠키 프라페", "descriptor": "맛있다", "sentiment": "긍정"}],
        "FACILITY": [{"entity": "콘센트", "descriptor": "많다", "sentiment": "긍정"}],
        "ATMOSPHERE": [],
        "TARGET": [{"entity": "카공", "descriptor": "좋다", "sentiment": "긍정"}]
    }, ensure_ascii=False)},
]

# ── STEP 1. 배치 입력 파일 생성 (수정됨) ───────────────────
def create_batch_input(df: pd.DataFrame, chunk_size: int = 2000) -> int:
    # 이미 완료된 항목 제외
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            done = set(json.load(f).keys())

    records = df.reset_index().to_dict("records")
    pending = [r for r in records if str(r["index"]) not in done]
    print(f"전체 대기 대상: {len(pending)}건 / 전체: {len(records)}건 (이미 완료: {len(done)}건)")

    if not pending:
        print("모두 완료된 상태입니다.")
        return 0

    # 토큰 한도를 넘지 않도록 남은 데이터 중 chunk_size 만큼만 잘라서 이번 배치에 할당
    chunk = pending[:chunk_size]
    print(f"이번 배치에 포함될 데이터: {len(chunk)}건")

    with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
        for item in chunk:
            request = {
                "custom_id": str(item["index"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "temperature": 0,
                    "max_tokens": 250,  # 1000에서 250으로 축소
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *FEW_SHOT,
                        {"role": "user", "content": item["리뷰내용"]}
                    ]
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"배치 파일 생성 완료 → {BATCH_INPUT_FILE}")
    return len(chunk)

# ── STEP 2. 배치 제출 ─────────────────────────────
def submit_batch() -> str:
    print("배치 파일 업로드 중...")
    batch_file = client.files.create(
        file=open(BATCH_INPUT_FILE, "rb"),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)

    print(f"배치 제출 완료!")
    print(f"  Batch ID : {batch.id}")
    print(f"  ID 저장  : {BATCH_ID_FILE}")
    print(f"  예상 완료: 수 분 ~ 수 시간 (최대 24h)")
    return batch.id

# ── STEP 3. 상태 확인 (수정됨) ─────────────────────────────
def check_status() -> str:
    with open(BATCH_ID_FILE, "r") as f:
        batch_id = f.read().strip()

    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"상태     : {batch.status}")
    print(f"진행률   : {counts.completed} 완료 / {counts.failed} 실패 / {counts.total} 전체")
    
    # 실패 시 상세 에러 원인 출력
    if batch.status == "failed" and batch.errors:
        print("\n[!] 배치 실패 원인 상세:")
        for error in batch.errors.data:
            print(f" - 코드: {error.code}")
            print(f" - 메시지: {error.message}")
            if hasattr(error, 'line'):
                print(f" - 발생 라인: {error.line}")
                
    return batch.status

# ── STEP 4. 결과 수집 ─────────────────────────────
def collect_results(df: pd.DataFrame) -> None:
    with open(BATCH_ID_FILE, "r") as f:
        batch_id = f.read().strip()

    batch = client.batches.retrieve(batch_id)
    print(f"배치 상태: {batch.status}")

    if batch.status != "completed":
        counts = batch.request_counts
        print(f"아직 완료되지 않았습니다.")
        print(f"진행률: {counts.completed} / {counts.total} (실패: {counts.failed})")
        return

    # 결과 파일 다운로드
    print("결과 다운로드 중...")
    result_content = client.files.content(batch.output_file_id).text
    lines = [json.loads(l) for l in result_content.strip().split("\n") if l.strip()]

    # 텍스트 맵 (index → 리뷰내용)
    text_map = dict(zip(df.index.astype(str), df['리뷰내용']))

    # 기존 결과와 병합
    results_dict = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results_dict = json.load(f)

    success, fail = 0, 0
    for line in lines:
        idx = line["custom_id"]
        if line.get("error"):
            print(f"  [Index {idx}] API 에러: {line['error']}")
            fail += 1
            continue
        try:
            content = line["response"]["body"]["choices"][0]["message"]["content"]
            ner_tags = json.loads(content)
            results_dict[idx] = {
                "index": int(idx),
                "original_text": text_map.get(idx, ""),
                "ner_tags": ner_tags
            }
            success += 1
        except Exception as e:
            print(f"  [Index {idx}] 파싱 에러: {e}")
            fail += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    print(f"\n수집 완료: 성공 {success}건 / 실패 {fail}건")
    print(f"저장 완료 → {OUTPUT_FILE}")


# ── 실행 ──────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(INPUT_FILE)

    # ① 2000건만 새롭게 묶어서 제출 (이 부분 주석 해제!)
    # count = create_batch_input(df, chunk_size=500)
    # if count > 0:
    #     submit_batch()

    # ② 상태 확인 (제출이 끝난 후 나중에 확인할 때만 주석 해제)
    # check_status()

    # ③ 완료 후 결과 수집 (completed 상태일 때만 주석 해제)
    collect_results(df)
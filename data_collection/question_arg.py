import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from prompt import question_system_prompt

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OUTPUT_FILE = "../data/cafe_questions.json"
BATCH_SIZE = 100   # 한 번에 100개씩 요청
TARGET_TOTAL = 3000
system_prompt = question_system_prompt

def generate_questions(n: int) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.2,   # 다양성 위해 높게 설정
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{n}개 생성해줘"}
        ],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    # 배열이 어떤 키로 감싸져도 꺼낼 수 있게
    if isinstance(parsed, list):
        return parsed
    return list(parsed.values())[0]

def main():
    all_questions = []

    # 기존 파일 있으면 이어서
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        print(f"기존 데이터 로드: {len(all_questions)}개")

    with tqdm(total=TARGET_TOTAL, initial=len(all_questions), desc="질문 생성") as pbar:
        while len(all_questions) < TARGET_TOTAL:
            remaining = TARGET_TOTAL - len(all_questions)
            batch = min(BATCH_SIZE, remaining)
            try:
                questions = generate_questions(batch)
                all_questions.extend(questions)

                # 중복 제거
                all_questions = list(dict.fromkeys(all_questions))

                # 즉시 저장
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_questions, f, ensure_ascii=False, indent=2)

                pbar.update(len(questions))

            except Exception as e:
                print(f"\n에러 발생: {e}, 재시도...")
                continue

    print(f"\n완료. 총 {len(all_questions)}개 → {OUTPUT_FILE}")

main()
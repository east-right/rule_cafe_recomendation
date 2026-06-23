"""
메뉴+비메뉴 복합 질문 증강 (임베딩 학습용).

실제 파이프라인은 질문을 통째로 RAG에 넣으므로, 메뉴 텍스트가 섞여도
비메뉴 rule을 찾도록 임베딩을 강건하게 학습시킨다.

방법: finetune_embd_query_pos.json의 일부 질문 앞에 메뉴 멘션만 붙이고
      정답(pos/title)은 비메뉴 rule 그대로 유지 → 메뉴를 노이즈로 무시하게 학습.

  "조용한 카페 추천해줘"  →  "아이스아메리카노 맛집인데 조용한 카페 추천해줘"  (gold: 조용함 동일)

run: build_embd_data.py 다음에 실행 (in-place append).
"""

import json
import random
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[2]
QP_PATH   = BASE_DIR / "data" / "finetune_embd_query_pos.json"

SEED      = 42
AUG_RATIO = 0.25   # 전체 쌍의 이 비율만큼 메뉴-복합 버전 추가

# menu_item.csv 상위 메뉴에서 추린 실제 메뉴명 (자연스러운 것만)
MENUS = [
    "아메리카노", "아이스아메리카노", "라떼", "바닐라라떼", "딸기라떼", "에스프레소",
    "아인슈페너", "밀크티", "에이드", "수박주스", "그릭요거트",
    "크로플", "마카롱", "소금빵", "휘낭시에", "케이크", "치즈케이크", "에그타르트",
    "와플", "스콘", "베이글", "쿠키", "빙수", "푸딩", "버터바",
]

TEMPLATES = [
    "{m} 맛집인데 {q}",
    "{m} 맛있고 {q}",
    "{m} 잘하는 곳 중에 {q}",
    "{m} 먹고 싶은데 {q}",
    "{q} 그리고 {m}도 맛있으면 좋겠어",
]


def main():
    data = json.load(open(QP_PATH, encoding="utf-8"))
    random.seed(SEED)

    n_aug   = int(len(data) * AUG_RATIO)
    sampled = random.sample(data, n_aug)

    aug = []
    for d in sampled:
        m = random.choice(MENUS)
        t = random.choice(TEMPLATES)
        aug.append({
            "query": t.format(m=m, q=d["query"]),
            "pos":   d["pos"],     # 비메뉴 rule description 그대로
            "title": d["title"],   # gold = 비메뉴 rule 그대로
        })

    data += aug
    json.dump(data, open(QP_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"메뉴-복합 증강 {len(aug)}개 추가 → 총 {len(data)}개")
    print("샘플:")
    for x in aug[:4]:
        print(f"  [{x['title']}] {x['query']}")


if __name__ == "__main__":
    main()

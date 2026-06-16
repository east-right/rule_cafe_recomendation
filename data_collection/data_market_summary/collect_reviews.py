"""
매장별 item 키워드 기반 리뷰 선별 스크립트

키워드 형식: "마카롱_가성비 있다" → entity("마카롱")로 리뷰 매칭
저장: data/market_review_summary.json
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = ROOT / "data" / "shinline_cafe_reviews_test.csv"
ITEMS_PATH = ROOT / "data" / "market_item.csv"
OUTPUT_PATH = ROOT / "data" / "market_review_summary.json"


def extract_entity(keyword: str) -> str:
    """'마카롱_가성비 있다' → '마카롱'"""
    return keyword.split("_")[0].strip()


def load_items() -> dict[str, list[str]]:
    """매장별 긍정 item 키워드의 entity 목록 반환"""
    store_entities: dict[str, set[str]] = defaultdict(set)
    with open(ITEMS_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["sentiment"] == "긍정" and row["최종_키워드"]:
                entity = extract_entity(row["최종_키워드"])
                store_entities[row["사업장명"]].add(entity)
    return {store: list(entities) for store, entities in store_entities.items()}


def load_reviews() -> dict[str, list[str]]:
    """매장별 리뷰 목록 반환"""
    store_reviews: dict[str, list[str]] = defaultdict(list)
    with open(REVIEWS_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["리뷰내용"].strip():
                store_reviews[row["사업장명"]].append(row["리뷰내용"].strip())
    return dict(store_reviews)


def match_reviews(
    entities: list[str],
    reviews: list[str],
    min_match: int = 2,
    max_reviews: int = 20,
) -> list[str]:
    """
    entity가 min_match개 이상 포함된 리뷰 우선 선택
    부족하면 1개 포함된 리뷰로 채워 max_reviews개 반환
    """
    scored = []
    for review in reviews:
        count = sum(1 for e in entities if e in review)
        if count > 0:
            scored.append((count, review))

    scored.sort(key=lambda x: x[0], reverse=True)

    priority = [r for c, r in scored if c >= min_match]
    fallback = [r for c, r in scored if c < min_match]

    selected = priority[:max_reviews]
    if len(selected) < max_reviews:
        selected += fallback[: max_reviews - len(selected)]

    return selected


def main():
    print("[INFO] item 키워드 로드 중...")
    store_entities = load_items()
    print(f"  매장 수: {len(store_entities)}")

    print("[INFO] 리뷰 로드 중...")
    store_reviews = load_reviews()
    print(f"  리뷰 있는 매장 수: {len(store_reviews)}")

    result = {}
    no_match_stores = []

    for store, entities in store_entities.items():
        reviews = store_reviews.get(store, [])
        matched = match_reviews(entities, reviews)

        if matched:
            result[store] = {
                "entities": entities,
                "review_count": len(reviews),
                "matched_count": len(matched),
                "reviews": matched,
            }
        else:
            # 키워드 매칭 실패 → 전체 리뷰를 fallback으로 사용
            fallback = reviews[:20]
            if fallback:
                result[store] = {
                    "entities": entities,
                    "review_count": len(reviews),
                    "matched_count": len(fallback),
                    "reviews": fallback,
                    "fallback": True,
                }
            else:
                no_match_stores.append(store)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fallback_count = sum(1 for v in result.values() if v.get("fallback"))
    print(f"\n[결과]")
    print(f"  총 매장: {len(result)}개 (키워드 매칭: {len(result) - fallback_count}개, fallback: {fallback_count}개)")
    print(f"  리뷰 없어 제외: {len(no_match_stores)}개")
    if no_match_stores:
        for s in no_match_stores:
            print(f"    - {s}")

    # 샘플 출력
    sample_store = next(iter(result))
    sample = result[sample_store]
    print(f"\n[샘플: {sample_store}]")
    print(f"  entities: {sample['entities'][:5]}")
    print(f"  전체 리뷰: {sample['review_count']}개 → 선별: {sample['matched_count']}개")
    preview = sample['reviews'][0][:80].encode('cp949', errors='replace').decode('cp949')
    print(f"  리뷰 예시: {preview}...")
    print(f"\n[INFO] 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

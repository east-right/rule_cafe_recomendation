"""리뷰 기반 LLM 채점 (골드 qrels) — 1단계 엄격 pointwise.

관련도 정의: '부합 리뷰가 존재하냐'가 아니라
             '그 요구가 이 카페의 실제 강점/사람들이 그걸 하러 찾는 특징(destination)이냐'.
스치는 언급 = 1점, 반복·중심 증거 = 3점, 반함·무관 = 0.

키워드추출/검색 없음(순환·커버리지 회피). gpt-4.1-mini, temp 0.
전 매장 × 그 매장 리뷰 전부. 실제 대량 채점은 run_judging.py(배치).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from openai import OpenAI

import batch_runner as br  # .env + stdout utf-8

HERE = Path(__file__).resolve().parent
ROOT = br.ROOT
MAX_REVIEWS = 150

RUBRIC = (
    "너는 카페 추천 평가자다. 사용자 질문과 특정 카페의 실제 리뷰 전부가 주어진다.\n"
    "이 카페가 이 질문의 요구에 대해 '실제로 추천할 만한 곳'인지 리뷰 증거로 판정하라.\n\n"
    "가장 중요한 원칙 — '존재'가 아니라 '대표성/강점'으로 본다:\n"
    "- 부합 리뷰가 하나 있다고 점수를 주지 마라.\n"
    "- 그 요구가 이 카페의 '실제 강점이자 사람들이 그걸 하러 찾는 특징'인지 봐라.\n"
    "  · 여러 리뷰가 반복적·중심적으로 그 요구를 지지 → 강점(destination)\n"
    "  · 스치듯 한두 번, 부수적으로 언급 → 가능은 하나 그 목적지는 아님\n"
    "- 예('공부하기 좋은 곳' 질문): '지하 스터디존/카공족 많음/콘센트' 반복 → 공부 목적지(높음).\n"
    "  디저트·분위기 카페인데 '조용해서 집중되네' 한두 줄 → 부수적(낮음).\n"
    "- 질문이 특정 대상(예: 딸기 케이크, 디카페인, 한강 뷰)을 콕 집으면 '바로 그 대상'의 직접 증거가 있어야 한다.\n"
    "  비슷한 것/상위 범주로 대체해서 점수 주지 마라. (딸기 케이크 질문에 치즈케이크·마카롱 맛집은 낮게.)\n\n"
    "점수:\n"
    " 3 = 이 요구가 이 카페의 명백한 대표 강점. 반복·중심 증거, 반하는 근거 거의 없음.\n"
    " 2 = 분명한 강점이나 '대표'까지는 아님.\n"
    " 1 = 가능은 하나 부수적/스침. 카페 정체성은 딴 데 있음.\n"
    " 0 = 무관하거나 반하는 근거가 우세.\n\n"
    "리뷰가 적으면(1~3개) 확신하지 말고 최대 1점.\n\n"
    'JSON만: {"grade":0-3, "is_strength":true/false, '
    '"evidence":["직접 근거 인용 최대 3"], "reason":"한 줄"}'
)


def s1_messages(question: str, reviews: list[str], total_n: int | None = None) -> list[dict]:
    joined = "\n".join(f"- {r}" for r in reviews)
    if total_n and total_n > len(reviews):
        head = f"리뷰(이 매장 총 {total_n}개 중 질문과 관련된 {len(reviews)}개만 표시):"
    else:
        head = f"리뷰({len(reviews)}개):"
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"질문: {question}\n\n{head}\n{joined}"},
    ]


def load_selection() -> dict:
    p = HERE / "data" / "review_selection.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def reviews_for(qid: str, cafe: str, groups: dict, selection: dict):
    """(보낼 리뷰 subset, 그 매장 총 리뷰수). 선택에 없으면(≤50) 전부."""
    all_r = groups[cafe]
    sel = selection.get(qid, {}).get(cafe)
    if sel is None:
        return all_r, len(all_r)
    return [all_r[i] for i in sel], len(all_r)


def judge_one(client: OpenAI, question: str, reviews: list[str],
              total_n: int | None = None) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=s1_messages(question, reviews, total_n),
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def load_reviews() -> dict[str, list[str]]:
    path = ROOT / "data" / "reviews_clean.csv"
    if not path.exists():
        path = ROOT / "data" / "shinline_cafe_reviews_test.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["리뷰내용"])
    return df.groupby("사업장명")["리뷰내용"].apply(list).to_dict()


def load_questions() -> list[dict]:
    # 평가용 축소본(50) 우선, 없으면 전체(153)
    for name in ("questions_eval.json", "questions.json"):
        p = HERE / "data" / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("questions*.json 없음")


def test_sample(qid: str = "q000", n: int = 15) -> None:
    client = OpenAI()
    groups = load_reviews()
    qmap = {q["qid"]: q["question"] for q in load_questions()}
    random.seed(0)
    cafes = random.sample(sorted(groups), n)
    question = qmap[qid]
    print(f"[{qid}] {question}")
    rows = [(judge_one(client, question, groups[c]).get("grade", 0), c) for c in cafes]
    for g, c in sorted(rows, reverse=True):
        print(f"  {g}점  {c}")


if __name__ == "__main__":
    test_sample()

"""평가셋 질문 생성 파이프라인 (zero-shot 배치 → dedup → 최종 데이터).

흐름:
  1) 소스 확보: 이미 생성된 questions.json 있으면 재사용, 없으면 배치로 생성.
     (신림동 맥락 8개 테마별 요청을 한 배치로)
  2) dedup: 글자(difflib) + 의미(임베딩 코사인) 유사 질문 제거.
  3) 최종 저장: eval_data/data/questions.json  (이것만 남김)
  4) 청소: 배치 인풋/아웃풋/batch_id 등 중간 파일 전부 삭제.

- src(테마)는 생성 다양성용 provenance일 뿐 채점엔 안 씀(채점은 리뷰 보고 holistic).
- 재실행 안전: data/questions.json 있으면 그냥 종료.
"""
from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from openai import OpenAI

import batch_runner as br

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT = DATA_DIR / "questions.json"
NAME = "questions"
PER_THEME = 12

THEMES = {
    "study":      "혼자 공부하거나 노트북으로 작업하려는 상황 (집중·콘센트·좌석·시험기간·밤늦게)",
    "meet":       "친구·연인·지인을 만나려는 상황 (수다·소개팅·모임·오랜만의 만남)",
    "solo_mood":  "혼자 조용히 시간 보내려는 상황 (멍때림·독서·사색·혼자만의 휴식)",
    "menu":       "특정 메뉴를 먹고 싶은 상황 (디저트·커피맛·차/논커피·빵·특정 음료)",
    "vibe":       "분위기/감성을 중시하는 상황 (인테리어·음악·아늑함·사진 찍기)",
    "practical":  "실용 조건을 따지는 상황 (가성비·넓은 좌석·영업시간·주차·테이크아웃)",
    "tradeoff":   "두세 조건이 서로 충돌하는 까다로운 요구 (예: 조용한데 너무 적막하진 않은)",
    "maybe_absent": "신림동에 없을 법한 것을 찾는 상황 (루프탑·한강뷰·대형 베이커리·반려동물 동반·비건 전문 등)",
}

RULES = (
    "규칙:\n"
    "- 진짜 구어체. 검색하듯/친구에게 묻듯 자연스럽게.\n"
    "- 말투·길이·끝맺음을 전부 다르게. '~카페', '~카페 추천' 같은 똑같은 패턴 반복 금지.\n"
    "- '카페'라는 단어가 아예 없는 질문도 섞어라.\n"
    "- 서로 비슷하거나 의도가 겹치는 질문은 피해라. 각각 달라야 한다.\n"
    'JSON만 출력: {"questions": ["...", "...", ...]}'
)

# 청소 대상(중간 파일)
JUNK = [
    HERE / f"{NAME}_input.jsonl",
    HERE / f"{NAME}_output.jsonl",
    HERE / f"{NAME}_batch_id.txt",
    HERE / f"{NAME}_errors.jsonl",
    HERE / f"{NAME}.json",  # 이전 top-level 산출물
]


# ---------- 1) 소스 확보 ----------
def make_prompt(desc: str) -> str:
    return (
        "너는 서울 관악구 신림동에서 갈 카페를 찾는 사람이다.\n"
        "신림동은 고시촌·서울대 근처라 학생, 자취생, 직장인이 많다.\n\n"
        f"상황: {desc}\n\n"
        f"이 상황에서 사람들이 카페를 찾을 때 실제로 던질 법한 질문을 {PER_THEME}개 만들어라.\n\n"
        + RULES
    )


def build_requests() -> list[dict]:
    return [
        br.build_request(
            custom_id=theme,
            messages=[{"role": "user", "content": make_prompt(desc)}],
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        for theme, desc in THEMES.items()
    ]


def generate_new_sync(client: OpenAI) -> list[dict]:
    """배치 큐가 느릴 때: 테마별 동기 호출로 즉시 생성(질문 생성은 소량이라 OK)."""
    raw: list[dict] = []
    for theme, desc in THEMES.items():
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": make_prompt(desc)}],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        try:
            qs = json.loads(resp.choices[0].message.content)["questions"]
        except Exception as e:
            print(f"⚠️ {theme}: 파싱 실패 {e}")
            continue
        cnt = 0
        for q in qs:
            q = q.strip()
            if q:
                raw.append({"question": q, "src": theme})
                cnt += 1
        print(f"  {theme}: {cnt}개")
    return raw


def parse_results(results: dict[str, dict]) -> list[dict]:
    raw: list[dict] = []
    for theme in THEMES:
        body = results.get(theme)
        if not body:
            print(f"⚠️ {theme}: 결과 없음")
            continue
        content = body["choices"][0]["message"]["content"]
        try:
            qs = json.loads(content)["questions"]
        except Exception as e:
            print(f"⚠️ {theme}: 파싱 실패 {e}")
            continue
        for q in qs:
            q = q.strip()
            if q:
                raw.append({"question": q, "src": theme})
    return raw


def load_existing() -> list[dict]:
    """이미 확정된 질문(data/questions.json)을 시드로 로드."""
    if not OUT.exists():
        return []
    data = json.loads(OUT.read_text(encoding="utf-8"))
    print(f"기존 질문 시드: {len(data)}개")
    return [{"question": d["question"], "src": d.get("src", "?")} for d in data]


def generate_new(client: OpenAI) -> list[dict]:
    """새 배치를 제출·수거해 질문 raw 리스트 반환(중단 시 resume 안전)."""
    out_jsonl = HERE / f"{NAME}_output.jsonl"
    if out_jsonl.exists():
        print(f"기존 배치 출력 재사용: {out_jsonl.name}")
        results = {}
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                results[o["custom_id"]] = o["response"]["body"]
        return parse_results(results)

    bid_file = HERE / f"{NAME}_batch_id.txt"
    if bid_file.exists():
        batch_id = bid_file.read_text(encoding="utf-8").strip()
        print(f"기존 배치 이어받음: {batch_id}")
    else:
        batch_id = br.submit(build_requests(), NAME, client=client)
    batch = br.poll(batch_id, client=client)
    if batch.status != "completed":
        print(f"아직 미완료(status={batch.status}). 다시 실행하면 이어받습니다.")
        sys.exit(0)
    return parse_results(br.fetch(batch, NAME, client=client))


# ---------- 2) dedup ----------
def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def lexical_dedup(items: list[dict], threshold: float = 0.80) -> list[dict]:
    kept: list[dict] = []
    for it in items:
        n = _norm(it["question"])
        if any(SequenceMatcher(None, n, _norm(k["question"])).ratio() >= threshold
               for k in kept):
            continue
        kept.append(it)
    return kept


def semantic_dedup(items: list[dict], client: OpenAI,
                   threshold: float = 0.80,
                   model: str = "text-embedding-3-small") -> list[dict]:
    resp = client.embeddings.create(model=model, input=[it["question"] for it in items])
    v = np.array([d.embedding for d in resp.data])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    kept: list[int] = []
    for i in range(len(items)):
        dup = next(((j, float(v[i] @ v[j])) for j in kept if float(v[i] @ v[j]) >= threshold),
                   None)
        if dup:
            j, sim = dup
            print(f"  drop  {items[i]['question']}\n     ↳ {sim:.2f} 유사  «{items[j]['question']}»")
        else:
            kept.append(i)
    return [items[i] for i in kept]


# ---------- 3~4) 저장 + 청소 ----------
def cleanup() -> None:
    for p in JUNK:
        if p.exists():
            p.unlink()


def main() -> None:
    client = OpenAI()
    existing = load_existing()      # 기존 확정분(시드, 먼저 => 우선 유지)
    new = generate_new_sync(client)  # 새 질문(동기 생성)
    items = existing + new
    print(f"\n합계 원본: {len(items)}개 (기존 {len(existing)} + 신규 {len(new)})")

    items = lexical_dedup(items)
    print(f"글자 dedup 후: {len(items)}개")

    print("의미 dedup:")
    items = semantic_dedup(items, client)
    print(f"의미 dedup 후: {len(items)}개\n")

    final = [{"qid": f"q{i:03d}", "question": it["question"], "src": it["src"]}
             for i, it in enumerate(items)]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    for it in final:
        print(f'{it["qid"]}  [{it["src"]:>12}]  {it["question"]}')
    print(f"\n최종 저장: {OUT}  ({len(final)}개)")

    cleanup()
    print("중간 파일 청소 완료.")


if __name__ == "__main__":
    main()

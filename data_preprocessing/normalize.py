"""
NER 정규화 파이프라인  ner_result_v2.json → ner_result_v3.json
[0] unique 추출
[1] 표기 정규화 (규칙)
[3] entity LLM 정규화 (카테고리별 정책)
[4] descriptor LLM 정규화
[5] 매핑 적용
"""

import json, re, sys, collections
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
client = OpenAI()

DATA   = ROOT / "data"
INPUT  = DATA / "ner_result_v2.json"
OUTPUT = DATA / "ner_result_v3.json"
MAP_DIR = DATA / "norm_mapping"
MAP_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 40


# ---------- [0] unique 추출 ----------

def extract_unique(data: dict):
    ec = collections.defaultdict(collections.Counter)
    dc = collections.defaultdict(collections.Counter)
    for entities in data.values():
        for e in entities:
            cat = e["category"]
            ec[cat][e["entity"]] += 1
            if e.get("descriptor") and e["type"] == "EVALUATIVE":
                dc[cat][e["descriptor"]] += 1
    return ec, dc


# ---------- [1] 표기 정규화 (규칙) ----------

def rule_norm(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------- [3][4] LLM 정규화 ----------

ENTITY_POLICY = {
"MENU": """\
규칙:
- 온도/강도/시즌 수식어 제거: 아이스아메리카노→아메리카노, 핫라떼→라떼
- 재료 수식어는 메뉴를 구분하면 유지: 바닐라라떼, 딸기라떼, 소금빵 유지
- 줄임말 확장: 아아→아메리카노, 아라→아이스라떼
- 동의어: 카페라떼→라떼, 치즈케익→치즈케이크
- 포괄어 → null: 커피, 음료, 디저트, 빵, 푸드, 주류, 메뉴, 원두, 크림
- 비메뉴 → null: 맛, 서비스, 가격, 분위기
- 디카페인: 그대로 유지""",

"FACILITY": """\
규칙:
- 좌석류 병합: 자리→좌석, 의자→좌석
- 공간류 병합: 내부→매장, 가게→매장, 실내→매장, 카페→매장, 매장 내부→매장
- 창/뷰: 창→창문, 창가자리→창문, 경치→뷰, 풍경→뷰, 전망→뷰
- 층 개별 유지: 1층, 2층, 3층, 지하 각각 그대로
- 변별력 있는 시설 유지: 통창, 테라스, 단체석, 콘센트, 포토존, 와이파이, 흡연실
- 포괄어 → null: 시설, 환경""",

"ATMOSPHERE": """\
규칙:
- 음악류 병합: 노래→음악, 배경음악→음악
- 감성류 병합: 갬성→감성, 인스타감성→감성
- 뷰류 병합: 경치→뷰, 풍경→뷰, 전망→뷰
- 형용사 누수 → 분위기: 조용함→분위기, 아늑함→분위기, 깔끔함→분위기, 깨끗함→분위기, 쾌적함→분위기
- 포괄어 → null: 느낌, 환경, 기분, 장소, 매장, 카페, 가게""",

"TARGET": """\
규칙:
- 혼자: 혼자방문→혼자, 혼밥→혼자
- 공부(작업과 반드시 분리): 공부하기→공부, 카공→공부, 집중하기→공부, 집중→공부
- 작업(공부와 반드시 분리): 작업하기→작업, 노트북작업→작업, 일하기→작업, 노트북하기→작업
- 대화: 수다→대화, 대화하기→대화
- 반려동물: 반려견동반→반려동물, 애견동반→반려동물, 강아지동반→반려동물, 반려동물동반→반려동물, 강아지→반려동물
- 데이트: 커플→데이트, 남자친구→데이트, 여자친구→데이트, 연인→데이트
- 포괄어 → null: 방문""",
}

DESC_POLICY = """\
규칙:
- 종결어미 통일 → ~다 형태: 좋은→좋다, 맛있어요→맛있다, 조용한→조용하다, 넓은→넓다
- 동의어: 이쁘다→예쁘다
- 청결류 통합: 깨끗하다→깔끔하다, 청결하다→깔끔하다
- 복합 → 주 형용사만: "넓고 쾌적하다"→넓다, "부드럽고 맛있다"→맛있다, "조용하고 아늑하다"→조용하다
- 극성 보존 (병합 금지): 맛있다≠맛없다, 조용하다≠시끄럽다, 넓다≠좁다, 깔끔하다≠지저분하다
- 의미 없는 descriptor → null: 그렇다, 그런 것 같다"""


def _llm_normalize_chunk(items: list, policy: str) -> dict:
    prompt = f"""아래 한국어 카페 리뷰 키워드/표현 목록을 정규화하세요.

{policy}

규칙에 해당 없으면 원본 그대로 반환. null은 집계에서 제외할 항목에만 사용.

입력:
{json.dumps(items, ensure_ascii=False)}

출력 (JSON):
{{"원본1": "정규화형 or null", "원본2": "정규화형 or null"}}"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def normalize_list(counter: collections.Counter, policy: str, label: str) -> dict:
    save_path = MAP_DIR / f"{label}.json"

    mapping: dict = {}
    if save_path.exists():
        mapping = json.load(open(save_path, encoding="utf-8"))

    items = [rule_norm(k) for k in counter if rule_norm(k) not in mapping]

    for i in tqdm(range(0, len(items), CHUNK_SIZE), desc=label, file=sys.stdout):
        chunk = items[i : i + CHUNK_SIZE]
        result = _llm_normalize_chunk(chunk, policy)
        mapping.update(result)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    return mapping


# ---------- [4.5] 공백-무시 병합 (후처리) ----------

def consolidate_spacing(mapping: dict, counter: collections.Counter) -> dict:
    """LLM 정규화 결과를 공백 제거 기준으로 그룹핑해, 빈도 최다 표면형으로 통합.
    공백만 다른 동일 표현(수박주스/수박 주스)을 안전하게 병합한다."""
    # rule_norm 기준 빈도
    rn_freq = collections.Counter()
    for k, v in counter.items():
        rn_freq[rule_norm(k)] += v

    # canonical별 누적 빈도 (해당 canonical로 매핑된 원본들의 빈도 합)
    canon_freq = collections.Counter()
    for orig, canon in mapping.items():
        if canon:
            canon_freq[canon] += rn_freq.get(orig, 1)

    # 공백 제거 키로 그룹핑 → 최빈 변형을 대표로
    groups = collections.defaultdict(list)
    for canon in canon_freq:
        groups[canon.replace(" ", "")].append(canon)

    winner = {}
    for variants in groups.values():
        best = max(variants, key=lambda c: canon_freq[c])
        for c in variants:
            winner[c] = best

    return {o: (None if c is None else winner.get(c, c)) for o, c in mapping.items()}


# ---------- [4.7] 개념 클러스터링 (비-MENU entity) ----------
# 전체 어휘를 한 번에 보고 "일관된 대표 키워드"로 통일한다.
# 주차/주차장/주차비 -> 주차장, 충전/플러그 -> 콘센트. 단 콘센트≠와이파이는 분리 유지.

CLUSTER_CATS = ["FACILITY", "ATMOSPHERE", "TARGET"]


def _post_entity_freq(counter: collections.Counter, entity_map: dict) -> collections.Counter:
    """entity 정규화(+공백병합) 적용 후 canonical 빈도."""
    c = collections.Counter()
    for k, v in counter.items():
        canon = entity_map.get(rule_norm(k), rule_norm(k))
        if canon:
            c[canon] += v
    return c


def assign_or_create(chunk: list, vocab: list, cat: str) -> dict:
    """온라인 클러스터링: 기존 개념(vocab)에 흡수하거나, 없으면 새 개념 생성."""
    vocab_str = json.dumps(vocab, ensure_ascii=False) if vocab else "(아직 없음)"
    prompt = f"""카페 리뷰의 '{cat}' 키워드를 일관된 개념으로 정리하는 중입니다.

[지금까지 만들어진 개념 목록]
{vocab_str}

[새 키워드들] 각각을 아래 규칙으로 처리하세요.
- 위 개념 목록에 **같은 의미**가 있으면 그 개념으로 매핑
- 없으면 **새 개념을 간결하게** 만들어 매핑 (그 키워드 자체를 대표로 쓰거나 더 간결한 형태로)

[합치기 - 적극적으로]
- 표기/변형/하위표현: 주차/주차비 -> 주차장, 충전/플러그 -> 콘센트, 책읽기/독서 -> 독서
- 흩어진 잔챙이를 개념으로: 아점/모닝커피/아침시간 -> 아침, 등산/따릉이/운동후 -> 운동·산책, 서울대안/서울대캠퍼스내 -> 서울대
- 동반자/행동: 남편/소개팅/커플링 -> 데이트, 카공/노트북/글쓰기 -> 공부

[합치지 말 것 - 변별 특징 보존]
- 서로 다른 구체 특징: 콘센트 != 와이파이, 1인좌석 != 단체석, 통창 != 창문
- 추상 상위개념(작업편의/시설/공간/목적)으로 뭉개지 말 것. 개념은 항상 구체적으로.

입력: {json.dumps(chunk, ensure_ascii=False)}
출력(JSON): {{"원본1": "개념", "원본2": "개념", ...}}  (모든 입력 키를 포함)"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def cluster_entities(canon_freq: collections.Counter, cat: str) -> dict:
    """빈도 높은 순으로 청크 처리. 빈출 특징이 먼저 anchor가 되고 잔챙이가 흡수된다."""
    map_path = MAP_DIR / f"cluster_{cat}.json"
    voc_path = MAP_DIR / f"vocab_{cat}.json"

    mapping = json.load(open(map_path, encoding="utf-8")) if map_path.exists() else {}
    # 기존 매핑에서 vocab 복원 (resume 대비)
    vocab = sorted(set(v for v in mapping.values() if v), key=lambda x: -canon_freq.get(x, 0))

    items = [k for k, _ in canon_freq.most_common() if k not in mapping]
    for i in tqdm(range(0, len(items), CHUNK_SIZE), desc=f"cluster_{cat}", file=sys.stdout):
        chunk = items[i : i + CHUNK_SIZE]
        res = assign_or_create(chunk, vocab, cat)
        for term, concept in res.items():
            if not concept:
                continue
            mapping[term] = concept
            if concept not in vocab:
                vocab.append(concept)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        with open(voc_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    return mapping


# ---------- [5] 매핑 적용 ----------

def apply_mapping(data: dict, entity_maps: dict, desc_maps: dict, cluster_maps: dict) -> dict:
    result = {}
    for place_id, entities in data.items():
        normalized = []
        for e in entities:
            cat = e["category"]

            orig_entity  = rule_norm(e["entity"])
            canon_entity = entity_maps[cat].get(orig_entity, orig_entity)
            if canon_entity is None:
                continue
            # 개념 클러스터링 적용 (비-MENU)
            canon_entity = cluster_maps.get(cat, {}).get(canon_entity, canon_entity)

            new_e = dict(e)
            new_e["entity"] = canon_entity

            if e.get("descriptor") and e["type"] == "EVALUATIVE":
                orig_desc   = rule_norm(e["descriptor"])
                canon_desc  = desc_maps.get(cat, {}).get(orig_desc, orig_desc)
                new_e["descriptor"] = None if canon_desc is None else canon_desc

            normalized.append(new_e)

        if normalized:
            result[place_id] = normalized
    return result


# ---------- main ----------

def main():
    data = json.load(open(INPUT, encoding="utf-8"))
    ec, dc = extract_unique(data)

    print("=== [0] unique 추출 ===")
    for cat in ["MENU", "FACILITY", "ATMOSPHERE", "TARGET"]:
        print(f"  {cat}: entity {len(ec[cat])} / desc {len(dc[cat])}")

    print("\n=== [3] entity 정규화 ===")
    entity_maps = {}
    for cat, policy in ENTITY_POLICY.items():
        m = normalize_list(ec[cat], policy, f"entity_{cat}")
        entity_maps[cat] = consolidate_spacing(m, ec[cat])

    print("\n=== [4] descriptor 정규화 ===")
    desc_maps = {}
    for cat in ["MENU", "FACILITY", "ATMOSPHERE"]:
        m = normalize_list(dc[cat], DESC_POLICY, f"desc_{cat}")
        desc_maps[cat] = consolidate_spacing(m, dc[cat])

    print("\n=== [4.7] 개념 클러스터링 (비-MENU) ===")
    cluster_maps = {}
    for cat in CLUSTER_CATS:
        cf = _post_entity_freq(ec[cat], entity_maps[cat])
        cluster_maps[cat] = cluster_entities(cf, cat)

    print("\n=== [5] 매핑 적용 ===")
    v3 = apply_mapping(data, entity_maps, desc_maps, cluster_maps)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(v3, f, ensure_ascii=False)

    all_v3 = [e for v in v3.values() for e in v]
    dropped = sum(len(v) for v in data.values()) - len(all_v3)
    print(f"완료: {len(v3)}개 카페 / {len(all_v3)}개 entity (null 제외 {dropped}개)")
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()

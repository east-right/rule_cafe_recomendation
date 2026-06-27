import json

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1  Entity extraction: category / type / sentiment / review_idx
# ─────────────────────────────────────────────────────────────────────────────

NER_PASS1_SYSTEM_PROMPT = """You are an expert NLP model specializing in Named Entity Recognition (NER) combined with sentiment classification for the Korean F&B and Cafe domain.

Your task is to analyze Korean customer reviews and extract entities, classify each into a category and a type, and assign sentiment polarity.

---

### [Entity Categories]

| Category   | Definition | Examples |
|------------|-----------|---------|
| MENU       | Specific food/beverage items, ingredients, options | 아메리카노, 크로플, 디카페인, 비건옵션 |
| FACILITY   | Physical structures, objects, spatial elements of the store | 콘센트, 테라스, 좌석, 화장실, 루프탑, 와이파이, 주차 |
| ATMOSPHERE | Mood/vibe described via abstract noun + evaluative adjective, OR a self-contained concept/style word | 분위기, 음악, 조명, 냄새 / 인스타감성, 빈티지컨셉, 24시간 |
| TARGET     | Visit purpose, companion type, demographic, social activity | 카공, 데이트, 반려동물동반, 단체모임, 혼자방문 |

---

### [Type Classification — EVALUATIVE vs FEATURE]

Every entity must be classified as one of two types:

**EVALUATIVE**: the review expresses a degree, quality, or state about the entity (how good/bad/much).
- Triggered by degree words: 많다, 적다, 편하다, 불편하다, 좋다, 깨끗하다, 조용하다, 시끄럽다, 비싸다, 맛있다 등
- This entity will receive a descriptor in a later pass.

**FEATURE**: the review only states whether the entity exists/is offered or not — no degree or quality judgment.
- Triggered by existence words: 있다, 없다, 가능하다, 제공하다, 운영하다, or simply naming something as present
- This entity will NOT receive a descriptor.

**Category-specific rules:**

1. **TARGET is ALWAYS FEATURE.** Visit purpose/companion/activity is never evaluated by degree — only whether the cafe suits that purpose. Discard all evaluative wrapper words entirely (좋다, 정말, 딱이다, 강추 등).
   - "카공하기 정말 좋아요" → entity: 카공, type: FEATURE
   - "혼자 오기 좋아요" → entity: 혼자방문, type: FEATURE

2. **MENU / FACILITY**: judge by whether a degree/quality word is present.
   - "콘센트가 많아요" → entity: 콘센트, type: EVALUATIVE (많다 = degree)
   - "콘센트가 있어요" → entity: 콘센트, type: FEATURE (있다 = existence only)
   - "디카페인이 맛있어요" → entity: 디카페인, type: EVALUATIVE
   - "디카페인 메뉴가 있어요" → entity: 디카페인, type: FEATURE

3. **ATMOSPHERE**: judge by entity form.
   - Abstract noun (분위기, 음악, 조명, 냄새) + evaluative adjective → EVALUATIVE
     - "분위기가 조용해요" → entity: 분위기, type: EVALUATIVE
   - Self-contained concept/style word that needs no further qualifier → FEATURE
     - "인스타 감성이에요" → entity: 인스타감성, type: FEATURE
     - "24시간 운영해요" → entity: 24시간, type: FEATURE

---

### [Slang Normalization]

Normalize Korean cafe slang and abbreviations to the full canonical name in the entity field. Never output slang — output the canonical name:

| Slang | Canonical |
|-------|-----------|
| 아아   | 아이스아메리카노 |
| 뜨아   | 아메리카노 |
| 아라   | 아이스라떼 |
| 아샤추 | 아이스샤인머스캣청 |
| 얼죽아 | 아이스아메리카노 |
| 바라   | 바닐라라떼 |

Apply broadly to any well-known Korean cafe slang or abbreviation you recognize.

---

### [Entity Extraction Rules]

**R1. Entity must be a strict noun or noun phrase**
- Forbidden: verb phrases, adjective predicates, full sentences/clauses as entity.
- If a mood is expressed with only a predicate and no noun, skip — do not fabricate a noun.

**R2. Single-category assignment**
Apply priority order:
1. Physical object/structure you can touch or enter → FACILITY
2. Abstract mood/vibe (not physical) or self-contained concept word → ATMOSPHERE
3. Visit purpose/companion/demographic/activity → TARGET
4. Food or drink item → MENU

**R3. Multiple distinct evaluations of the same entity → separate objects**

**R4. Skip staff/service mentions entirely**
- 직원, 사장님, 친절, 서비스 — never extract.

**R5. Existence-negation counts as FEATURE with negative sentiment**
- "흡연실이 없어요" → entity: 흡연실, type: FEATURE, sentiment: 부정
- "24시간이 아니라서 아쉬워요" → entity: 24시간, type: FEATURE, sentiment: 부정

**R6. Sentiment**
- 긍정: approval, satisfaction, recommendation.
- 부정: dissatisfaction, complaint, warning, or absence of a desired feature.
- 중립: purely factual mention with no clear polarity.

**R7. review_idx**
- Each entity must include the 1-based index of the review it was extracted from, matching the [리뷰 N] label in the input.

---

### [Output Schema]

Return a JSON object with four keys: MENU, FACILITY, ATMOSPHERE, TARGET — each a list of entity objects. Empty list [] if none found.

Each entity object:
{
  "entity": "<noun or noun phrase — slang normalized>",
  "type": "EVALUATIVE" | "FEATURE",
  "sentiment": "긍정" | "부정" | "중립",
  "review_idx": <integer, 1-based index of source review>
}

Output strictly in JSON. No explanation, no extra text.
"""

PASS1_FEW_SHOT: list[dict] = [
    # Example 1: basic + slang normalization + EVALUATIVE/FEATURE + skip staff
    {
        "role": "user",
        "content": (
            "[리뷰 1] 루프탑 뷰가 너무 좋고 크로플이 맛있어요. 데이트 코스로 강추!\n"
            "[리뷰 2] 아아 한 잔 시켰는데 양이 너무 적어요. 콘센트가 있어서 카공하기 좋아요.\n"
            "[리뷰 3] 주차장이 없어서 많이 불편했어요. 사장님은 정말 친절하셨어요."
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "MENU": [
                {"entity": "크로플",          "type": "EVALUATIVE", "sentiment": "긍정", "review_idx": 1},
                {"entity": "아이스아메리카노", "type": "EVALUATIVE", "sentiment": "부정", "review_idx": 2}
            ],
            "FACILITY": [
                {"entity": "루프탑",  "type": "EVALUATIVE", "sentiment": "긍정", "review_idx": 1},
                {"entity": "콘센트",  "type": "FEATURE",    "sentiment": "긍정", "review_idx": 2},
                {"entity": "주차장",  "type": "FEATURE",    "sentiment": "부정", "review_idx": 3}
            ],
            "ATMOSPHERE": [],
            "TARGET": [
                {"entity": "데이트", "type": "FEATURE", "sentiment": "긍정", "review_idx": 1},
                {"entity": "카공",   "type": "FEATURE", "sentiment": "긍정", "review_idx": 2}
            ]
        }, ensure_ascii=False)
    },

    # Example 2: ATMOSPHERE EVALUATIVE vs FEATURE + FEATURE negation + skip service
    {
        "role": "user",
        "content": (
            "[리뷰 1] 분위기가 정말 조용하고 아늑해요. 인스타 감성이 넘쳐요.\n"
            "[리뷰 2] 24시간 운영이라 편해요. 화장실이 너무 지저분해요. 직원 서비스는 불친절해요.\n"
            "[리뷰 3] 흡연실이 없어요. 반려견 동반 가능해서 좋았어요."
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "MENU": [],
            "FACILITY": [
                {"entity": "화장실", "type": "EVALUATIVE", "sentiment": "부정", "review_idx": 2},
                {"entity": "흡연실", "type": "FEATURE",    "sentiment": "부정", "review_idx": 3}
            ],
            "ATMOSPHERE": [
                {"entity": "분위기",    "type": "EVALUATIVE", "sentiment": "긍정", "review_idx": 1},
                {"entity": "인스타감성", "type": "FEATURE",    "sentiment": "긍정", "review_idx": 1},
                {"entity": "24시간",    "type": "FEATURE",    "sentiment": "긍정", "review_idx": 2}
            ],
            "TARGET": [
                {"entity": "반려견동반", "type": "FEATURE", "sentiment": "긍정", "review_idx": 3}
            ]
        }, ensure_ascii=False)
    },

    # Example 3: same entity multiple evaluations + slang + FACILITY EVALUATIVE vs FEATURE
    {
        "role": "user",
        "content": (
            "[리뷰 1] 아메리카노는 맛은 좋은데 양이 너무 적어요. 수다 떨기 좋아요.\n"
            "[리뷰 2] 내부가 넓고 콘센트도 충분해요. 디카페인 메뉴가 있어요.\n"
            "[리뷰 3] 아샤추가 달아서 별로였어요. 테라스가 있고 음악이 너무 시끄러워요."
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "MENU": [
                {"entity": "아메리카노",          "type": "EVALUATIVE", "sentiment": "긍정", "review_idx": 1},
                {"entity": "아메리카노",          "type": "EVALUATIVE", "sentiment": "부정", "review_idx": 1},
                {"entity": "디카페인",            "type": "FEATURE",    "sentiment": "긍정", "review_idx": 2},
                {"entity": "아이스샤인머스캣청",  "type": "EVALUATIVE", "sentiment": "부정", "review_idx": 3}
            ],
            "FACILITY": [
                {"entity": "내부",   "type": "EVALUATIVE", "sentiment": "긍정", "review_idx": 2},
                {"entity": "콘센트", "type": "EVALUATIVE", "sentiment": "긍정", "review_idx": 2},
                {"entity": "테라스", "type": "FEATURE",    "sentiment": "긍정", "review_idx": 3}
            ],
            "ATMOSPHERE": [
                {"entity": "음악", "type": "EVALUATIVE", "sentiment": "부정", "review_idx": 3}
            ],
            "TARGET": [
                {"entity": "수다", "type": "FEATURE", "sentiment": "긍정", "review_idx": 1}
            ]
        }, ensure_ascii=False)
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2  Descriptor extraction for EVALUATIVE entities
# ─────────────────────────────────────────────────────────────────────────────

NER_PASS2_SYSTEM_PROMPT = """You are an expert in Korean Aspect-Based Sentiment Analysis (ABSA) for the cafe domain.

You will receive a numbered list of EVALUATIVE entities extracted from cafe reviews. Each entry includes the entity, its category, its pre-confirmed sentiment, and the source review text it was extracted from.

Your task: write a concise Korean "descriptor" for each entity, and assess your confidence.

---

### [Descriptor Rules]

R1. Strip bare intensifiers (너무, 정말, 진짜, 매우, 엄청, 넘, 꽤, 좀) when they only amplify a simple adjective.
  - "너무 좋다" → "좋다" / "엄청 맛있다" → "맛있다"

R2. Concessive clause (~지만, ~긴 하지만, ~는데): keep ONLY the final conclusive evaluation.
  - "달긴 하지만 맛있다" → "맛있다" / "비싸지만 양이 많다" → "양이 많다"

R3. Abstract symptom/evidence to evaluative conclusion.
  - "벌레가 엄청나게 많다" → "청결하지 않다"
  - "에어컨이 안 나온다" → "덥다"

R4. Do NOT re-judge sentiment — the sentiment is already confirmed. Use it only as context.

R5. Descriptor must be a standalone predicate phrase — do NOT include the entity name.
  - entity: "분위기" → "조용하다" (O) / "분위기가 조용하다" (X)
  - entity: "콘센트" → "많다" (O) / "콘센트가 많다" (X)

R6. If no descriptor can be determined from the source review, return null.

---

### [Confidence Rules]

Every descriptor must be assigned a confidence value — "확신" or "애매".

**확신**: The descriptor is directly and clearly supported by the source review for this specific entity. No ambiguity.

**애매**: Mark as 애매 in ANY of these cases:
  - The source review contains strongly mixed signals (positive + negative for different aspects) making it hard to isolate this entity's evaluation
  - You cannot confidently determine what the descriptor should be
  - The natural descriptor would require including the entity name to be meaningful

---

### [Output Schema]

Return a JSON object with key "descriptors". Use the same idx numbers as the input list.

{
  "descriptors": [
    {"idx": <integer>, "descriptor": "<Korean predicate phrase>" | null, "confidence": "확신" | "애매"},
    ...
  ]
}

Output strictly in JSON. No explanation, no extra text.
"""

PASS2_FEW_SHOT: list[dict] = [
    # Example 1: 확신 케이스 — 강도 부사 제거 + entity명 descriptor 제외
    {
        "role": "user",
        "content": (
            "EVALUATIVE 엔티티:\n"
            "1. entity: \"루프탑\", category: FACILITY, sentiment: 긍정\n"
            "   원문: \"루프탑 뷰가 너무 좋고 크로플이 정말 맛있어요.\"\n"
            "2. entity: \"크로플\", category: MENU, sentiment: 긍정\n"
            "   원문: \"루프탑 뷰가 너무 좋고 크로플이 정말 맛있어요.\"\n"
            "3. entity: \"아이스아메리카노\", category: MENU, sentiment: 부정\n"
            "   원문: \"아이스아메리카노를 시켰는데 양이 너무 적어요.\"\n"
            "4. entity: \"내부\", category: FACILITY, sentiment: 긍정\n"
            "   원문: \"내부가 엄청 넓어요.\""
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "descriptors": [
                {"idx": 1, "descriptor": "뷰가 좋다",  "confidence": "확신"},
                {"idx": 2, "descriptor": "맛있다",     "confidence": "확신"},
                {"idx": 3, "descriptor": "양이 적다",  "confidence": "확신"},
                {"idx": 4, "descriptor": "넓다",       "confidence": "확신"}
            ]
        }, ensure_ascii=False)
    },

    # Example 2: 증상 추상화 + 양보절 + 동일 entity 복수 평가 + 애매 케이스
    {
        "role": "user",
        "content": (
            "EVALUATIVE 엔티티:\n"
            "1. entity: \"화장실\", category: FACILITY, sentiment: 부정\n"
            "   원문: \"화장실에 벌레가 엄청나게 많아요.\"\n"
            "2. entity: \"분위기\", category: ATMOSPHERE, sentiment: 긍정\n"
            "   원문: \"분위기가 정말 조용하고 아늑해요.\"\n"
            "3. entity: \"아메리카노\", category: MENU, sentiment: 긍정\n"
            "   원문: \"아메리카노가 달긴 하지만 맛있어요. 양은 너무 적고요.\"\n"
            "4. entity: \"아메리카노\", category: MENU, sentiment: 부정\n"
            "   원문: \"아메리카노가 달긴 하지만 맛있어요. 양은 너무 적고요.\"\n"
            "5. entity: \"음악\", category: ATMOSPHERE, sentiment: 부정\n"
            "   원문: \"음악 소리가 너무 커서 대화가 힘들었어요.\"\n"
            "6. entity: \"디저트\", category: MENU, sentiment: 긍정\n"
            "   원문: \"디저트 자체 맛은 괜찮았는데 매장 관리가 아예 안되고 있는듯 자리에 앉아있는 내내 벌레가 엄청나게 많아요.\""
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "descriptors": [
                {"idx": 1, "descriptor": "청결하지 않다",    "confidence": "확신"},
                {"idx": 2, "descriptor": "조용하고 아늑하다", "confidence": "확신"},
                {"idx": 3, "descriptor": "맛있다",           "confidence": "확신"},
                {"idx": 4, "descriptor": "양이 적다",        "confidence": "확신"},
                {"idx": 5, "descriptor": "시끄럽다",         "confidence": "확신"},
                {"idx": 6, "descriptor": "맛이 괜찮다",      "confidence": "애매"}
            ]
        }, ensure_ascii=False)
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Response format schemas (OpenAI structured outputs)
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_ITEM = {
    "type": "object",
    "properties": {
        "entity":     {"type": "string"},
        "type":       {"type": "string", "enum": ["EVALUATIVE", "FEATURE"]},
        "sentiment":  {"type": "string", "enum": ["긍정", "부정", "중립"]},
        "review_idx": {"type": "integer"}
    },
    "required": ["entity", "type", "sentiment", "review_idx"],
    "additionalProperties": False
}

PASS1_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ner_pass1_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "MENU":       {"type": "array", "items": _ENTITY_ITEM},
                "FACILITY":   {"type": "array", "items": _ENTITY_ITEM},
                "ATMOSPHERE": {"type": "array", "items": _ENTITY_ITEM},
                "TARGET":     {"type": "array", "items": _ENTITY_ITEM}
            },
            "required": ["MENU", "FACILITY", "ATMOSPHERE", "TARGET"],
            "additionalProperties": False
        }
    }
}

PASS2_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ner_pass2_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "descriptors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idx":        {"type": "integer"},
                            "descriptor": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "confidence": {"type": "string", "enum": ["확신", "애매"]}
                        },
                        "required": ["idx", "descriptor", "confidence"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["descriptors"],
            "additionalProperties": False
        }
    }
}

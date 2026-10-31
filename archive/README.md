# archive/ — 보류·미사용 자료 보관소

> 2026-10-29 정리. **삭제가 아니라 이동**입니다. 최종 LLM 추천 서비스(`service/` + `inference/`)가
> 런타임에 쓰지 않는 코드/자료를, 기록 보존을 위해 여기로 옮겨 모았습니다.
> 되살리려면 `git mv archive/<경로> <원래경로>` 로 복구하면 됩니다.

## 왜 옮겼나 (판단 기준)

- 최종 서비스 코드가 **import/참조하지 않는 것**만 이동.
- 데이터 생성 파이프라인(`data_preprocessing/`, `embedding/`, `keyword_selection/`,
  `question_classifier/`, `data_collection/`)과 그 산출물은 **건드리지 않음** — 모델·DB의 재료이자
  고군분투 기록이라 보존.

## 폴더별 내용

### `soar/` — Soar 트랙 (최종 로직 미사용)
현재 추천 추론은 `service/nodes/soar_recommend.py`의 **"Soar 영감 Python trace"** 이고,
**진짜 Soar는 서비스에서 쓰지 않는다.** 아래는 정식 Soar 시도의 흔적:
- `sml/` — Soar C++ SML 벤더링(약 148MB, **git 미추적** / `.gitignore`). track2(KG+GNN) 재개 시 참고.
- `Make_Rule/` — `soar_rule_keywords.json` → `.soar` 파일 생성기(`generate_rule.py`, `soar_rules/*.soar`).
- `LLM_keyword/` — LLM 키워드 추출 실험.
- `check_sample.py` — `.soar` 샘플 1회용 생성 스크립트.
- ⚠️ **주의**: `data/soar_rule_keywords.json` 은 옮기지 **않았다** — 서비스가 OpenSearch 룰 키워드
  적재에 실제로 사용한다(`keyword_selection/uodate_index.py`).

### `ml/` — ML 랭킹 트랙 (사용 안 하기로 결정)
- `ml_ranking_poc.ipynb` — LambdaMART POC 노트북.
- `README.md` — 구 `notebooks/README.md`.
- `pools.json`, `query_set.json` — ML/eval POC 중간 데이터(코드 참조 0). `data/qrels.csv`는
  LLM 서비스 평가에도 쓰므로 `data/`에 **남겨둠**.

### `misc_unused/` — 죽은 코드 / 스크래치
- `queries.py` — 구 `service/db/queries.py`. 어떤 그래프 노드도 import 안 함(죽은 코드).
- `notepade.ipynb` — 구 `keyword_selection/` 스크래치.
- `notepad.ipynb` — 구 `question_classifier/` 스크래치.

## 옮기지 않은 것 (의도적으로 보존)
- `data_preprocessing/` 전체(+`archive/`의 GLiNER·v1 실험) — 상점 item 생성 고군분투.
- `embedding/`의 hybrid·rerank 실험, `question_classifier/`의 koelectra·roberta 파인튜닝 —
  결과가 안 나왔거나 교체됐어도 **의사결정 근거/베이스라인**이라 보존.
- `eval_data/` — gold qrels는 LLM 서비스 평가 자산(ML 전용 아님).

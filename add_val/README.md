# Cafe XAI 감성 분리 — 0·1단계 작업 패키지

기준: `east-right/rule_cafe_recomendation` `main` commit **`0ba0670b9a84791ce2d459c3f8fa03d2080d41ef`**.

GitHub 앱의 읽기 권한은 있지만 `create_branch`가 403으로 거부되어, **GitHub에는 어떤 파일도 수정하지 않았습니다**. 이 패키지는 원본에 적용할 스크립트와 검증 코드입니다. 대상 파일이 바뀌었거나 이미 수정된 경우 자동 적용을 거부합니다.

## 포함 파일

- `apply_sentiment_fix.py`: `service/nodes/soar_recommend.py`와 `eval_data/eval_pipeline.py`에 동일한 감성 분리 적용. 정확한 기존 Git blob SHA 검사, 구문 검사, 소스 백업, 작업 중 오류 복구 포함.
- `snapshot_replay.py`: 기존 `pipeline_eval_results.json`의 **고정된 Rule 선택 결과**와 읽기 전용 `data/cafe.db`로 이전/수정 후 Trace와 Top-3를 재생하는 **순수 오프라인** 스냅샷. 원본 DB·파일 해시 기록.
- `tests/test_sentiment_matching.py`: 감성 충돌(부정만 있는 주차장, 긍정·부정 동시 존재, Tiebreak, negative narrowing, 메뉴 후보 제한 등) 단위 테스트. 별도 LLM, OpenSearch 없이 실제 함수의 AST를 추출해 검사.

## 로컬 실행 순서 — 반드시 기준선 먼저

1. 저장소를 클론/업데이트하고 **작업용 브랜치** 생성. 기준 커밋은 상단 값을 확인합니다. 모든 아래 예시는 저장소 밖에서 이 패키지 파일을 실행한다고 가정합니다.

```bash
# 저장소 내부에서 (이미 클론했다면 git clone 생략)
git status --short
git rev-parse HEAD
# 본인 브랜치에서만 수행
git switch -c fix/sentiment-matching
```

2. **기준선** 생성. 모델 서버·OpenSearch 없어도 됩니다. 생성 JSON은 절대 덮어쓰지 않도록 설계되어 있습니다.

```bash
python /path/to/package/snapshot_replay.py --repo /path/to/repo --mode baseline --out /path/to/repo/artifacts/baselines/before_sentiment.json
```

3. 변경 내용 검증(실제 수정 안 함) 후 적용:

```bash
python /path/to/package/apply_sentiment_fix.py --repo /path/to/repo
python /path/to/package/apply_sentiment_fix.py --repo /path/to/repo --apply
```

4. 수정 후 단위 테스트 및 스냅샷:

```bash
CAFE_REPO_ROOT=/path/to/repo python -m unittest discover -s /path/to/package/tests -v
python /path/to/package/snapshot_replay.py --repo /path/to/repo --mode corrected --out /path/to/repo/artifacts/baselines/after_sentiment.json
python /path/to/package/snapshot_replay.py --repo /path/to/repo --mode compare --out /path/to/repo/artifacts/baselines/paired_comparison.json
```

Windows PowerShell에서는 테스트 실행 전에 `$env:CAFE_REPO_ROOT="C:\path\to\repo"`를 설정하세요.

5. 결과를 확인한 뒤 Git diff와 테스트를 검토하고 본인 브랜치에 커밋하세요. 운영 OpenSearch 인덱스는 변경하지 않습니다.

## 중요한 범위 제한

- **이번 패키지는 0번(고정 Rule 기반선)·1번(서비스/오프라인 평가의 감성 분리)만 다룹니다.** 답변 근거 혼입, Impasse 영구 저장, menu-complex 병렬 분기, Top-10/15 실험은 아직 수정되지 않았습니다.
- 오프라인 `snapshot_replay.py`는 **실제 전체 서비스 실행 로그가 아닙니다**. 기존 평가 JSON에서 이미 선택된 Rule을 재사용하므로 Rule Retrieval/Selection, 메뉴 분기, Impasse LLM, Tier 1/2 Fallback, 최종 LLM 답변을 재실행하지 않습니다. Trace는 2026-10-31 기준 서비스 함수의 로직을 재현합니다.
- `eval_pipeline.py`는 서비스의 후보 축소 알고리즘이 아니라 **점수 기반 Top-3 랭킹**입니다. 감성을 수정해도 이 차이는 유지됩니다. 평가 수치를 최종 서비스 Hit@1이라고 주장하면 안 됩니다.
- 부정 키워드가 관측되지 않았다는 사실은 **그 문제가 실제로 없음을 증명하지 않습니다**.
- `data/cafe.db` 직접 실행에 기반한 집계 결과는 **사용자가 패키지를 실행한 후 생성된 JSON로만 확정**하세요. 이전 CSV 집계 41/32/15를 새로운 성과 수치로 사용하지 마세요.
- `artifacts/baselines/pre_sentiment_fix_sources`에는 패치 직전 두 소스의 사본이 생성됩니다. 버전 관리 시 본인 판단으로 포함하세요.

## 검증 완료 기준

두 파일의 패치 적용과 AST 문법 검사, 회귀 단위 테스트, 기준선 전후 비교가 모두 통과해야 1단계 완료입니다. 평가 모델을 재호출한 새 성과 수치까지 만들어지는 것은 아닙니다.

### 선택: 실제 서비스 Trace의 기준선 캡처

`--mode baseline` 오프라인 재생은 **평가 JSON에 고정된 Rule을 쓰는 근사 재생**이므로 실제 그래프 실행 상태가 아닙니다. 모델 서버와 OpenSearch가 준비된 로컬 환경에서는 아래로 **실제 LangGraph 상태/Trace**를 캡처하세요. 이 스크립트는 알려진 두 OpenSearch 쓰기 함수(`_persist_rule`, `_update_os_tiebreak`)를 가로채서 저장을 막습니다. 모델과 OpenAI API 호출은 그대로 실행되어 호출 비용이 발생할 수 있으며, Langfuse 관측 기록이 남을 수 있습니다.

```bash
# 반드시 패치 전, 기존 main에서 먼저 실행
python /path/to/package/snapshot_live_graph.py --repo /path/to/repo --out /path/to/repo/artifacts/baselines/live_before.json
# 패치 후, 다른 파일명으로 실행
python /path/to/package/snapshot_live_graph.py --repo /path/to/repo --out /path/to/repo/artifacts/baselines/live_after.json
```

이 스크립트는 **최종 답변 생성은 실행하지 않으므로** 사용자에게 스트리밍되는 자연어 답변은 저장하지 않습니다. 그 부분은 이후 XAI 근거 분리 단계에서 별도로 시험해야 합니다. 또한 모델 버전 및 OpenSearch 인덱스가 동일하지 않으면 before/after 차이를 감성 수정만의 효과로 단정할 수 없습니다.

**기존 평가 결과 보존:** `snapshot_replay.py`가 만들어내는 JSON에는 기존 `pipeline_eval_results.json`의 **전체 내용**을 `original_eval_results_copy`로 포함합니다. 따라서 이후 `eval_pipeline.py`를 재실행하여 원본 결과 파일이 덮어써지더라도 비교할 기준값은 별도로 남습니다.

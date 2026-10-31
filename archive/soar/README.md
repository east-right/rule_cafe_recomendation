# soar/ — 룰 생성 & 정식 Soar 트랙

룰 정의를 만들고(키워드 추출·`.soar` 파일 생성), 정식 Soar 인지 아키텍처 실험을 담는 모듈.

> **현재 서비스는 정식 Soar를 쓰지 않는다.** 실서비스(`service/soar_recommend.py`)는
> Soar에서 *영감받은* Python 규칙기반 trace다. 이 폴더는 (1) 룰 데이터 생성 도구와
> (2) 정식 Soar(SML) 실험(트랙2, KG+GNN 확장의 발판)을 보관한다.

## 하위 폴더

| 폴더 | 역할 |
|---|---|
| `LLM_keyword/` | `rule_metadata_merged.json` → LLM으로 rule별 operator/tiebreak/negative 키워드 추출 → `soar_rule_keywords.json` |
| `Make_Rule/` | `soar_rule_keywords.json` → `.soar` 프로덕션 룰 파일 생성 (`generate_rule.py`) |
| `sml/` | Soar SML 바이너리/바인딩 *(로컬 설치, .gitignore)* |

각 하위 폴더 상세는 해당 README 참조.

## 데이터 흐름

```
rule_metadata_merged.json ──LLM_keyword──▶ soar_rule_keywords.json ──Make_Rule──▶ *.soar
                                                   │
                                                   └──▶ (service·eval에서 룰 키워드로 사용)
```

## 상태

- 룰 키워드 생성(`LLM_keyword`)은 현행 파이프라인이 쓰는 `soar_rule_keywords.json`의 소스 → **유효**
- 정식 Soar(SML) 실행은 **트랙2(보류)** — soar-sml pip 동작은 검증됨, 나중에 KG+GNN 구조적 XAI 확장 시 참고

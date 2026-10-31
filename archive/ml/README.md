# notebooks/ — 실험 노트북

## `ml_ranking_poc.ipynb` — ML 랭킹 POC

룰/Soar 기반에서 **학습된 랭커(LambdaMART) + SHAP 설명**으로 방향 전환을 검증한 POC.

흐름:
1. `cafe.db` → 카페 아이템 데이터 (긍정키워드·메뉴·리뷰요약 → BGE-M3 임베딩 + 구조화 피처)
2. `valid_questions.json` → 질문셋 → 후보 풀링 → **LLM-as-judge**로 관련도 qrels 생성
3. 피처(임베딩 코사인·키워드겹침·부정히트 등) → 학습행렬
4. 베이스라인 NDCG@5 vs **LightGBM lambdarank** 비교
5. **SHAP**으로 피처 기여도(faithful 설명) 확인

> ⚠️ 이 노트북의 qrels(`data/qrels.csv`)는 **옛 키워드-질문 기반**이다.
> 이후 전매장×전리뷰 홀리스틱 LLM-judge 방식으로 재설계됨 → `eval_data/`(새 골드 qrels)와
> `eval_data/eval_compare.py`(룰 vs ML 공정 비교)가 이 POC의 후속.

관련 계획: `docs/ml-ranking-plan.md` *(로컬 전용, .gitignore)*

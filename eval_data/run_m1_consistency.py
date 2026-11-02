"""M1 — 결과 안정성: 같은 의도 변형에 같은 Rule/매장이 나오는가 + 반복 일치 + LLM 직접추천 비교.

입력: results/M1/questions.json (10 base + 40 변형 = 50).
  (A) 메인: 50문항 1회 → 그룹(5개)별 Rule/최종매장 일치율
  (B) 반복: 10 base ×3회 → 같은 질문 반복 시 Rule/매장 일치율
  (C) LLM 직접추천: 181곳 목록에서 gpt-4.1이 질문당 1곳 직접 선택 → 그룹별 매장 일치율(규칙기반과 비교)
선행: OpenSearch + 모델서버(:8001). 러너는 run_service_eval 의 측정 패치(langfuse off, 영구쓰기 차단) 재사용.
"""
from __future__ import annotations
import json, sqlite3, subprocess, sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval_data"))
sys.path.insert(0, str(ROOT))
import run_service_eval as rse          # 측정 패치 포함(그래프·run_question 재사용)
from openai import OpenAI
import os

QFILE = ROOT / "eval_data" / "results" / "M1" / "questions.json"
OUT = ROOT / "eval_data" / "results" / "M1"
DB = ROOT / "data" / "cafe.db"
_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def cafe_names() -> list[str]:
    conn = sqlite3.connect(str(DB))
    names = [r[0] for r in conn.execute("SELECT DISTINCT cafe_name FROM cafe_keywords ORDER BY cafe_name")]
    conn.close()
    return names


def run_rule(qtext: str) -> dict:
    r = rse.run_question({"qid": "m1", "question": qtext, "src": "m1"})
    return {"rule": r.get("selected_rule"), "cafe": (r.get("soar_result") or {}).get("cafe")}


def llm_direct(qtext: str, names: list[str]) -> str:
    sysmsg = ("당신은 서울 신림동 카페 추천기입니다. 아래 카페 목록에서 질문에 가장 맞는 카페 1곳의 "
              "이름을 목록 표기 그대로 정확히 한 줄로만 출력하세요. 설명 금지.")
    resp = _oai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": sysmsg},
                  {"role": "user", "content": f"[질문]\n{qtext}\n\n[카페 목록]\n" + "\n".join(names)}],
        max_tokens=40, temperature=0,
    )
    out = resp.choices[0].message.content.strip()
    return out if out in names else f"(목록밖:{out[:20]})"


def majority_frac(xs: list) -> float:
    xs = [x for x in xs]
    if not xs:
        return 0.0
    return Counter(xs).most_common(1)[0][1] / len(xs)


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    data = json.loads(QFILE.read_text(encoding="utf-8"))
    groups = data["groups"]
    names = cafe_names()
    print(f"[M1] groups={len(groups)} names={len(names)}")

    # (A) 메인 50 + (C) LLM 직접
    group_rows = []
    for g in groups:
        rule_res, llm_res = [], []
        for it in g["items"]:
            rr = run_rule(it["q"])
            ld = llm_direct(it["q"], names)
            rule_res.append({"id": it["id"], "q": it["q"], **rr, "llm_direct": ld})
            llm_res.append(ld)
            print(f"  [{g['intent']:14s}] {it['id']:8s} rule={rr['rule']} cafe={rr['cafe']} | llm={ld}")
        rules = [x["rule"] for x in rule_res]
        cafes = [x["cafe"] for x in rule_res]
        group_rows.append({
            "intent": g["intent"], "base_qid": g["base_qid"],
            "rule_based": {"rules": rules, "cafes": cafes,
                           "rule_majority_frac": round(majority_frac(rules), 3),
                           "cafe_majority_frac": round(majority_frac(cafes), 3),
                           "unique_rules": len(set(rules)), "unique_cafes": len(set(cafes))},
            "llm_direct": {"cafes": llm_res,
                           "cafe_majority_frac": round(majority_frac(llm_res), 3),
                           "unique_cafes": len(set(llm_res))},
            "items": rule_res,
        })

    # (B) 반복: 각 그룹 base 3회
    repeat = []
    for g in groups:
        base_q = g["items"][0]["q"]
        runs = [run_rule(base_q) for _ in range(3)]
        rset = {x["rule"] for x in runs}; cset = {x["cafe"] for x in runs}
        repeat.append({"intent": g["intent"], "base_q": base_q,
                       "runs": runs, "rule_all_same": len(rset) == 1, "cafe_all_same": len(cset) == 1})
        print(f"  [repeat {g['intent']:14s}] rule_same={len(rset)==1} cafe_same={len(cset)==1}")

    rb_rule = mean(x["rule_based"]["rule_majority_frac"] for x in group_rows)
    rb_cafe = mean(x["rule_based"]["cafe_majority_frac"] for x in group_rows)
    ld_cafe = mean(x["llm_direct"]["cafe_majority_frac"] for x in group_rows)
    rep_rule = mean(1.0 if x["rule_all_same"] else 0.0 for x in repeat)
    rep_cafe = mean(1.0 if x["cafe_all_same"] else 0.0 for x in repeat)

    doc = {"task": "M1", "commit": head, "n_groups": len(groups), "n_questions": sum(len(g["items"]) for g in groups),
           "note": "그룹=같은 의도(base+변형4). majority_frac=그룹 5개 중 최빈 Rule/매장 비율(1.0=완전일치). 반복=base 3회 모두 동일 여부. LLM직접=gpt-4.1이 181곳에서 1곳.",
           "summary": {
               "rule_based_same_rule_rate": round(rb_rule, 3),
               "rule_based_same_cafe_rate": round(rb_cafe, 3),
               "llm_direct_same_cafe_rate": round(ld_cafe, 3),
               "repeat_rule_all_same_rate": round(rep_rule, 3),
               "repeat_cafe_all_same_rate": round(rep_cafe, 3)},
           "groups": group_rows, "repeat": repeat}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "consistency.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== M1 요약 ====")
    print(f"  규칙기반 같은의도 Rule 일치율: {rb_rule:.3f}")
    print(f"  규칙기반 같은의도 매장 일치율: {rb_cafe:.3f}")
    print(f"  LLM직접  같은의도 매장 일치율: {ld_cafe:.3f}")
    print(f"  반복(3회) Rule 완전일치율: {rep_rule:.3f}  매장: {rep_cafe:.3f}")
    print(f"  저장: {OUT/'consistency.json'} (commit {head[:7]})")


if __name__ == "__main__":
    main()

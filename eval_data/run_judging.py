"""전체 채점 배치 오케스트레이션 (1단계 엄격 pointwise).

  질문 153 × 매장 205 = (질문,매장) 전수 → 배치로 채점 → qrels.csv

사용:
  python run_judging.py dry [n_questions]     # 제출 안 함: 요청수·용량·샘플만
  python run_judging.py submit [n_questions]  # 배치 제출(청크), batch id 저장
  python run_judging.py fetch                 # 완료된 배치 수거 → qrels.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

import batch_runner as br
import judge

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
IDS = HERE / "judge_batch_ids.json"
CAFES = DATA / "cafes.json"
QRELS = DATA / "qrels.csv"
CHUNK = 2050  # 배치당 요청 상한 ≈ 질문 10개 × 매장 205 → 5배치(작게 나눠 안전·중간fetch)


def build(limit: int | None):
    groups = judge.load_reviews()
    selection = judge.load_selection()
    cafes = sorted(groups)
    qs = judge.load_questions()
    if limit:
        qs = qs[:limit]
    reqs = []
    for q in qs:
        for ci, cafe in enumerate(cafes):
            revs, total = judge.reviews_for(q["qid"], cafe, groups, selection)
            reqs.append(br.build_request(
                custom_id=f"{q['qid']}##{ci}",
                messages=judge.s1_messages(q["question"], revs, total),
                model="gpt-4.1-mini",
                response_format={"type": "json_object"},
                temperature=0,
            ))
    return reqs, cafes, len(qs)


def cmd_dry(limit):
    reqs, cafes, nq = build(limit)
    size = sum(len(json.dumps(r, ensure_ascii=False)) for r in reqs)
    print(f"질문 {nq} × 매장 {len(cafes)} = 요청 {len(reqs):,}건")
    print(f"입력 파일 크기(추정): {size/1e6:.1f} MB  → 배치 {(-(-len(reqs)//CHUNK))}개")
    print(f"\n--- 샘플 요청[0] custom_id={reqs[0]['custom_id']} ---")
    print(reqs[0]["body"]["messages"][1]["content"][:300], "...")


def cmd_submit(limit):
    reqs, cafes, nq = build(limit)
    DATA.mkdir(parents=True, exist_ok=True)
    CAFES.write_text(json.dumps(cafes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"요청 {len(reqs):,}건 → 청크 제출")
    ids = []
    for i in range(0, len(reqs), CHUNK):
        name = f"judge_{i // CHUNK}"
        bid = br.submit(reqs[i:i + CHUNK], name)
        ids.append({"name": name, "id": bid})
    IDS.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"배치 {len(ids)}개 제출 완료. (fetch로 수거)")


def cmd_fetch():
    client = OpenAI()
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    cafes = json.loads(CAFES.read_text(encoding="utf-8"))

    # 전 배치 상태 먼저 점검
    batches = {it["name"]: client.batches.retrieve(it["id"]) for it in ids}
    done = sum(b.status == "completed" for b in batches.values())
    for name, b in batches.items():
        c = b.request_counts
        print(f"  {name}: {b.status}  ({c.completed}/{c.total})" if c else f"  {name}: {b.status}")
    if done < len(ids):
        print(f"\n완료 {done}/{len(ids)} — 모두 끝나면 다시 fetch(전체 완료 시 qrels 생성).")
        return

    rows = []
    for it in ids:
        res = br.fetch(batches[it["name"]], it["name"], client=client)
        for cid, body in res.items():
            qid, ci = cid.split("##")
            try:
                d = json.loads(body["choices"][0]["message"]["content"])
            except Exception:
                d = {"grade": 0, "reason": "parse_fail", "evidence": []}
            rows.append({
                "qid": qid, "cafe": cafes[int(ci)],
                "grade": d.get("grade", 0), "is_strength": d.get("is_strength", ""),
                "reason": d.get("reason", ""), "evidence": " | ".join(d.get("evidence", [])),
            })
    df = pd.DataFrame(rows).sort_values(["qid", "grade"], ascending=[True, False])
    df.to_csv(QRELS, index=False, encoding="utf-8-sig")
    print(f"qrels 저장: {QRELS}  ({len(df):,}행)")
    print(df["grade"].value_counts().sort_index().to_string())


def cmd_retry():
    """실패한 배치만 재제출 후 judge_batch_ids.json 갱신."""
    client = OpenAI()
    ids = json.loads(IDS.read_text(encoding="utf-8"))

    failed_names = []
    for it in ids:
        b = client.batches.retrieve(it["id"])
        status = b.status
        c = b.request_counts
        count = f"({c.completed}/{c.total})" if c else ""
        mark = "재제출" if status == "failed" else "유지  "
        print(f"  {mark}: {it['name']}  {status}  {count}")
        if status == "failed":
            failed_names.append(it["name"])

    if not failed_names:
        print("실패한 배치 없음.")
        return

    reqs, cafes, _ = build(None)
    id_map = {it["name"]: it["id"] for it in ids}
    for name in failed_names:
        i = int(name.split("_")[1])
        chunk_reqs = reqs[i * CHUNK:(i + 1) * CHUNK]
        bid = br.submit(chunk_reqs, name)
        id_map[name] = bid

    new_ids = [{"name": it["name"], "id": id_map[it["name"]]} for it in ids]
    IDS.write_text(json.dumps(new_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(failed_names)}개 재제출 완료. 완료 후 fetch.")


def cmd_sample(n_cafes=8):
    """배치 없이 동기로 소규모 채점 → qrels 형식 확인 (qrels_sample.csv)."""
    import random
    client = OpenAI()
    groups = judge.load_reviews()
    selection = judge.load_selection()
    cafes = sorted(groups)
    random.seed(0)
    pick = random.sample(cafes, n_cafes)
    qs = judge.load_questions()[:2]

    rows = []
    for q in qs:
        for cafe in pick:
            revs, total = judge.reviews_for(q["qid"], cafe, groups, selection)
            d = judge.judge_one(client, q["question"], revs, total)
            rows.append({
                "qid": q["qid"], "cafe": cafe, "grade": d.get("grade", 0),
                "is_strength": d.get("is_strength", ""),
                "reason": d.get("reason", ""), "evidence": " | ".join(d.get("evidence", [])),
            })
    df = pd.DataFrame(rows).sort_values(["qid", "grade"], ascending=[True, False])
    df.to_csv(DATA / "qrels_sample.csv", index=False, encoding="utf-8-sig")

    for q in qs:
        print(f"\n[{q['qid']}] {q['question']}")
        for _, r in df[df.qid == q["qid"]].iterrows():
            print(f"  {r.grade}점 | {str(r.is_strength):5} | {r.cafe[:18]:18} | {r.reason[:44]}")
    print(f"\n컬럼: {list(df.columns)}")
    print(f"저장: {DATA / 'qrels_sample.csv'}  ({len(df)}행)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dry"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "dry":
        cmd_dry(lim)
    elif cmd == "submit":
        cmd_submit(lim)
    elif cmd == "fetch":
        cmd_fetch()
    elif cmd == "retry":
        cmd_retry()
    elif cmd == "sample":
        cmd_sample(lim or 8)
    else:
        print("dry | submit [n] | fetch | sample [n_cafes]")

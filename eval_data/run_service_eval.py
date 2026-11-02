"""M0 기준선 러너 — 실제 서비스 LangGraph를 돌려 문항별 결과를 고정 저장.

측정용 하네스일 뿐, service/ · eval_pipeline 의 로직은 수정하지 않는다.
비침습 계측:
  - Langfuse OFF : import 전에 LANGFUSE_TRACING_ENABLED=false → @observe no-op (M3 지연 노이즈 제거)
  - M2 토큰      : openai Completions.create 를 하네스에서 래핑해 노드별 호출·토큰 로깅
                   (비스트리밍=정확 usage, 스트리밍 답변=tiktoken 추정)
  - M3 지연      : graph.stream(updates) 로 노드별 경과 + 전체 + 답변 첫 토큰 시간

주의: 답변 생성은 temperature=0.7 → 답변 텍스트는 재실행마다 달라진다(룰 경로는 거의 결정적).
      이 baseline 의 답변은 'B2 수정 전' 스냅샷으로 M6 에서 재사용한다.

사용:
  uv run python eval_data/run_service_eval.py --dry 2     # 스모크
  uv run python eval_data/run_service_eval.py --dry 10    # dry
  uv run python eval_data/run_service_eval.py             # 전체 50
선행: OpenSearch(:9200, cafe_rules) + 모델서버(inference.server :8001) 기동.
"""
from __future__ import annotations

import os

# ── 반드시 service/langfuse import 보다 먼저 ──
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval_data" / "data"
OUTDIR = ROOT / "eval_data" / "results" / "M0"
QUESTIONS = DATA / "questions_eval.json"

# ── 토큰 카운터 (gpt-4.1 계열 = o200k_base) ──
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")
    def ntok(s: str) -> int:
        return len(_ENC.encode(s or ""))
    TOK_METHOD = "tiktoken:o200k_base"
except Exception:
    def ntok(s: str) -> int:
        return max(1, len(s or "") // 2)  # 한글 대략치
    TOK_METHOD = "approx:len/2"


def msgs_tokens(messages: list[dict]) -> int:
    return sum(ntok(m.get("content", "")) + 4 for m in messages) + 3


# ── M2: OpenAI create 래핑 (노드 태깅은 콜스택 파일명으로) ──
NODE_FILES = {
    "classify.py": "question_valid",      # question_classifier/classify.py
    "menu_extract.py": "menu_extract",
    "rule_fallback.py": "rule_fallback",
    "rule_create.py": "rule_create",
    "impasse_resolve.py": "impasse_resolve",
    "answer_generate.py": "answer_generate",
}
_CALLS: list[dict] = []  # 현재 문항의 호출 로그


def _which_node() -> str:
    for fr in inspect.stack():
        node = NODE_FILES.get(os.path.basename(fr.filename))
        if node:
            return node
    return "unknown"


def _install_patch():
    """service import(=langfuse.openai 계측) 보다 먼저 호출해야 langfuse 가 이 래퍼를 감싼다."""
    import openai

    orig = openai.resources.chat.completions.Completions.create

    def wrapper(self, *a, **kw):
        rec = {
            "node": _which_node(),
            "model": kw.get("model", "?"),
            "stream": bool(kw.get("stream")),
            "in_tok": msgs_tokens(kw.get("messages", [])),
            "out_tok": None,
            "exact": False,
        }
        _CALLS.append(rec)
        resp = orig(self, *a, **kw)
        if not rec["stream"]:
            try:
                u = resp.usage
                rec["in_tok"], rec["out_tok"], rec["exact"] = (
                    u.prompt_tokens, u.completion_tokens, True,
                )
            except Exception:
                try:
                    rec["out_tok"] = ntok(resp.choices[0].message.content)
                except Exception:
                    rec["out_tok"] = 0
        return resp

    openai.resources.chat.completions.Completions.create = wrapper


_install_patch()

# ── 이제 서비스 import (스크립트가 eval_data/ 에서 실행돼 루트가 path 에 없음) ──
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from service.graph import build_graph          # noqa: E402
from service.nodes import answer_generate      # noqa: E402
from service.nodes import impasse_resolve      # noqa: E402
from service.nodes import rule_create          # noqa: E402

# 측정 무결성: 측정 중 OpenSearch 인덱스를 변형하는 두 영구쓰기를 차단.
#   (1) impasse 의 tiebreak append (B3 미수정 동작)
#   (2) rule_create(Tier2)의 신규 생성룰 persist (CogRec 온라인학습)
# 둘 다 '문항 내부 추론/추천'은 in-memory 로 그대로 수행되고, 저장 side-effect 만 막는다.
# → cafe_rules 오염 없음 + 문항 순서 의존 제거 + 재실행 재현성 확보.
impasse_resolve._update_os_tiebreak = lambda *a, **k: None
rule_create._persist_rule = lambda *a, **k: None

_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return round(s[i], 4)


def run_question(q: dict) -> dict:
    _CALLS.clear()
    rec: dict = {"qid": q["qid"], "question": q["question"], "src": q.get("src")}
    node_latency: dict[str, float] = {}
    order: list[str] = []
    state: dict = {"question": q["question"]}  # 그래프 입력 채널 + 노드 delta 누적

    t0 = time.perf_counter()
    try:
        t_prev = t0
        for update in graph().stream({"question": q["question"]}, stream_mode="updates"):
            now = time.perf_counter()
            dt = now - t_prev
            for node, delta in update.items():
                node_latency[node] = node_latency.get(node, 0.0) + dt
                order.append(node)
                if isinstance(delta, dict):
                    state.update(delta)
            t_prev = now
        graph_total = time.perf_counter() - t0

        # 답변 생성 (첫 토큰 시간 포함)
        ans_t0 = time.perf_counter()
        first_tok_t = None
        parts: list[str] = []
        for tok in answer_generate.stream(state):
            if first_tok_t is None:
                first_tok_t = time.perf_counter()
            parts.append(tok)
        answer_text = "".join(parts)
        answer_total = time.perf_counter() - ans_t0

        # 스트리밍 답변 호출의 out_tok 를 실제 답변으로 채움
        for c in _CALLS:
            if c["node"] == "answer_generate" and c["stream"] and c["out_tok"] is None:
                c["out_tok"] = ntok(answer_text)

        soar = state.get("soar_result") or {}
        rec.update({
            "question_type": state.get("question_type"),
            "selected_rule": state.get("selected_rule"),
            "operator_keywords": state.get("operator_keywords", []),
            "tiebreak_keywords": state.get("tiebreak_keywords", []),
            "negative_keywords": state.get("negative_keywords", []),
            "rule_fallback_used": bool(state.get("rule_fallback_used")),
            "fallback_rules": state.get("fallback_rules", []),
            "needs_rule_creation": bool(state.get("needs_rule_creation")),
            "rule_created": bool(state.get("rule_created")),
            "soar_result": {
                "type": soar.get("type"),
                "cafe": soar.get("cafe"),
                "candidates": soar.get("candidates", []),
                "trace": soar.get("trace", []),
            },
            "impasse_iterations": state.get("impasse_iterations", 0),
            "answer": answer_text,
            "node_order": order,
            "latency": {
                "nodes_s": {k: round(v, 4) for k, v in node_latency.items()},
                "graph_total_s": round(graph_total, 4),
                "answer_first_token_s": round(first_tok_t - ans_t0, 4) if first_tok_t else None,
                "answer_total_s": round(answer_total, 4),
                "end_to_end_s": round(graph_total + answer_total, 4),
            },
            "openai_calls": [dict(c) for c in _CALLS],
        })
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["openai_calls"] = [dict(c) for c in _CALLS]
    return rec


def summarize(results: list[dict]) -> dict:
    by_node: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    for r in results:
        for c in r.get("openai_calls", []):
            for agg, key in ((by_node, c["node"]), (by_model, c["model"])):
                d = agg.setdefault(key, {"calls": 0, "in_tok": 0, "out_tok": 0, "exact_calls": 0})
                d["calls"] += 1
                d["in_tok"] += c["in_tok"] or 0
                d["out_tok"] += c["out_tok"] or 0
                d["exact_calls"] += int(c["exact"])
    total_calls = sum(d["calls"] for d in by_node.values())
    total_in = sum(d["in_tok"] for d in by_node.values())
    total_out = sum(d["out_tok"] for d in by_node.values())

    # 지연 분포
    e2e = [r["latency"]["end_to_end_s"] for r in results if "latency" in r]
    graph_t = [r["latency"]["graph_total_s"] for r in results if "latency" in r]
    ftt = [r["latency"]["answer_first_token_s"] for r in results
           if r.get("latency", {}).get("answer_first_token_s") is not None]
    node_lat: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.get("latency", {}).get("nodes_s", {}).items():
            node_lat.setdefault(k, []).append(v)

    return {
        "openai": {
            "total_calls": total_calls, "total_in_tok": total_in, "total_out_tok": total_out,
            "by_node": by_node, "by_model": by_model,
        },
        "latency": {
            "end_to_end_s": {"mean": round(mean(e2e), 4) if e2e else 0,
                             "p50": pctl(e2e, 50), "p95": pctl(e2e, 95)},
            "graph_total_s": {"mean": round(mean(graph_t), 4) if graph_t else 0,
                              "p50": pctl(graph_t, 50), "p95": pctl(graph_t, 95)},
            "answer_first_token_s": {"mean": round(mean(ftt), 4) if ftt else 0,
                                     "p50": pctl(ftt, 50), "p95": pctl(ftt, 95)},
            "by_node_mean_s": {k: round(mean(v), 4) for k, v in node_lat.items()},
        },
    }


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main(dry: int | None, qids: list[str] | None = None, outfile: str | None = None,
         task: str = "M0"):
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if qids:
        want = set(qids)
        questions = [q for q in questions if q["qid"] in want]
    elif dry:
        questions = questions[:dry]
    print(f"[{task}] {len(questions)}문항 실행 (token={TOK_METHOD}, langfuse_tracing=off)\n")

    results = []
    for i, q in enumerate(questions, 1):
        r = run_question(q)
        results.append(r)
        calls = len(r.get("openai_calls", []))
        err = r.get("error")
        rule = r.get("selected_rule")
        cafe = (r.get("soar_result") or {}).get("cafe")
        e2e = r.get("latency", {}).get("end_to_end_s")
        print(f"  [{i:2d}/{len(questions)}] {r['qid']} type={r.get('question_type')} "
              f"rule={rule} cafe={cafe} calls={calls} e2e={e2e}s"
              + (f"  ERROR={err}" if err else ""))

    summary = summarize(results)
    doc = {
        "meta": {
            "task": task, "commit": git_head(),
            "n_questions": len(questions), "dry": dry, "qids": qids,
            "token_method": TOK_METHOD, "langfuse_tracing": "off",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "caveat": "서비스 경로 실측. 답변 temperature=0.7 → 답변 텍스트는 재실행마다 변동(룰 경로는 거의 결정적). "
                      "이 baseline 답변은 B2 수정 전 스냅샷으로 M6 에서 재사용.",
        },
        "summary": summary,
        "results": results,
    }
    if outfile:
        out = Path(outfile)
    else:
        suffix = f"_dry{dry}" if dry else ""
        out = OUTDIR / f"baseline{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    oi = summary["openai"]
    print(f"\n{'='*64}")
    print(f"  OpenAI 총 {oi['total_calls']}콜  in={oi['total_in_tok']}tok  out={oi['total_out_tok']}tok")
    for model, d in oi["by_model"].items():
        print(f"    {model:16s} calls={d['calls']:3d} in={d['in_tok']:7d} out={d['out_tok']:6d} (exact {d['exact_calls']})")
    lat = summary["latency"]
    print(f"  지연 e2e mean={lat['end_to_end_s']['mean']}s p50={lat['end_to_end_s']['p50']}s p95={lat['end_to_end_s']['p95']}s")
    print(f"  첫토큰 mean={lat['answer_first_token_s']['mean']}s")
    print(f"  저장: {out}  (commit {doc['meta']['commit'][:7]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, default=None, help="앞 N문항만 실행")
    ap.add_argument("--qids", type=str, default=None, help="쉼표구분 qid만 실행 (회귀용)")
    ap.add_argument("--outfile", type=str, default=None, help="출력 경로 지정")
    ap.add_argument("--task", type=str, default="M0", help="메타 task 라벨")
    args = ap.parse_args()
    qids = [q.strip() for q in args.qids.split(",")] if args.qids else None
    main(dry=args.dry, qids=qids, outfile=args.outfile, task=args.task)

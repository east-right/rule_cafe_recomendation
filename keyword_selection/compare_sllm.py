"""Rule 선택 정확도 모델별 비교 (389문항, test_with_candidates.jsonl).

같은 SYSTEM_PROMPT + build_user_prompt(keyword_selection/prompt.py) 사용, 모델만 교체.
메모리 안전을 위해 **모델 1개당 별도 프로세스**로 실행한다(1.5B 2개가 동시에 뜨면 OOM).

  uv run python keyword_selection/compare_sllm.py --model gpt      [--limit N]
  uv run python keyword_selection/compare_sllm.py --model gguf
  uv run python keyword_selection/compare_sllm.py --model base
  uv run python keyword_selection/compare_sllm.py --model adapter
  uv run python keyword_selection/compare_sllm.py --summarize      # 4개 합쳐 results.json

모델:
  gpt     : gpt-4.1-mini (temperature 0)            — OpenAI API, 389콜
  gguf    : qwen1.5b-cafe-q8_0.gguf (llama.cpp CPU) — 파인튜닝 GGUF
  base    : Qwen/Qwen2.5-1.5B-Instruct (원본)        — transformers CPU, 파인튜닝 없음
  adapter : base + east-right LoRA 어댑터            — transformers+peft CPU, 파인튜닝

지표: 전체 Accuracy, none Accuracy, match Accuracy, 건당 평균 지연, (gpt)건당 비용,
      후보 title 밖 출력(out_of_candidates) 건수.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # prompt.py
from prompt import SYSTEM_PROMPT, build_user_prompt   # noqa: E402

TEST = HERE / "data" / "test_with_candidates.jsonl"
GGUF = HERE / "qwen1.5b-cafe-q8_0.gguf"
BASE_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "east-right/cafe-keyword-selection-qwen-1.5b"
OUTDIR = ROOT / "eval_data" / "results" / "sllm_compare"
MAX_NEW = 32
GPT_IN_RATE, GPT_OUT_RATE = 0.40, 1.60   # per 1M (gpt-4.1-mini, 2026-11 docs)


def load_data(limit=None):
    rows = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    return rows[:limit] if limit else rows


def git_head():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# ── 예측기: (pred, latency, extra) ─────────────────────────────

def predictor_gpt():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    usage = {"in": 0, "out": 0}

    def predict(query, candidates):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, candidates)}]
        t = time.time()
        r = client.chat.completions.create(model="gpt-4.1-mini", messages=msgs,
                                           max_tokens=MAX_NEW, temperature=0)
        dt = time.time() - t
        usage["in"] += r.usage.prompt_tokens
        usage["out"] += r.usage.completion_tokens
        return r.choices[0].message.content.strip(), dt
    return predict, {"model": "gpt-4.1-mini", "usage": usage}


def predictor_gguf():
    from llama_cpp import Llama
    llm = Llama(model_path=str(GGUF), n_ctx=4096, n_threads=os.cpu_count(), verbose=False)

    def predict(query, candidates):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, candidates)}]
        t = time.time()
        r = llm.create_chat_completion(messages=msgs, max_tokens=MAX_NEW,
                                       temperature=0.0, repeat_penalty=1.1)
        dt = time.time() - t
        return r["choices"][0]["message"]["content"].strip(), dt
    return predict, {"model": "qwen1.5b-cafe-q8_0.gguf (llama.cpp CPU)"}


def _hf_predict(model_name, with_adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 4)
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[os.getenv("CMP_DTYPE", "bf16")]
    print(f"[hf] dtype={dtype} threads={torch.get_num_threads()} cores={os.cpu_count()}", flush=True)
    tok_src = ADAPTER_REPO if with_adapter else BASE_REPO
    tok = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, dtype=dtype, low_cpu_mem_usage=True)
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_REPO)
    model.eval()
    eos = tok.eos_token_id

    def predict(query, candidates):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, candidates)}]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        n_in = enc["input_ids"].shape[-1]
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=eos)
        dt = time.time() - t
        gen = out[0][n_in:]
        return tok.decode(gen, skip_special_tokens=True).strip(), dt
    return predict, {"model": model_name}


def run_model(model_key, limit):
    data = load_data(limit)
    if model_key == "gpt":
        predict, meta = predictor_gpt()
    elif model_key == "gguf":
        predict, meta = predictor_gguf()
    elif model_key == "base":
        predict, meta = _hf_predict("Qwen2.5-1.5B-Instruct (원본, transformers CPU)", False)
    elif model_key == "adapter":
        predict, meta = _hf_predict("Qwen2.5-1.5B + LoRA 어댑터 (transformers CPU)", True)
    else:
        raise SystemExit("unknown model")

    total = correct = none_total = none_correct = match_total = match_correct = 0
    out_of_cand = 0
    times = []
    details = []
    for i, d in enumerate(data, 1):
        titles = {c["title"] for c in d["candidates"]}
        pred, dt = predict(d["query"], d["candidates"])
        label = d["answer"]
        ok = pred == label
        in_cand = (pred == "none") or (pred in titles)
        total += 1; times.append(dt)
        if ok: correct += 1
        if not in_cand: out_of_cand += 1
        if label == "none":
            none_total += 1; none_correct += int(ok)
        else:
            match_total += 1; match_correct += int(ok)
        details.append({"query": d["query"], "answer": label, "pred": pred,
                        "correct": ok, "in_candidates": in_cand, "latency": round(dt, 4)})
        if i % 25 == 0 or i == len(data):
            print(f"  [{model_key}] {i}/{len(data)} acc={correct/total:.3f} ooc={out_of_cand}", flush=True)

    res = {
        "model_key": model_key, "model": meta["model"], "commit": git_head(),
        "n": total, "accuracy": round(correct / total, 4),
        "correct": correct,
        "none_total": none_total, "none_correct": none_correct,
        "none_accuracy": round(none_correct / none_total, 4) if none_total else None,
        "match_total": match_total, "match_correct": match_correct,
        "match_accuracy": round(match_correct / match_total, 4) if match_total else None,
        "out_of_candidates": out_of_cand,
        "avg_latency_s": round(mean(times), 4), "total_latency_s": round(sum(times), 2),
        "details": details,
    }
    if model_key == "gpt":
        u = meta["usage"]
        cost = u["in"] * GPT_IN_RATE / 1e6 + u["out"] * GPT_OUT_RATE / 1e6
        res["tokens"] = u
        res["cost_usd_total"] = round(cost, 5)
        res["cost_usd_per_req"] = round(cost / total, 7)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{model_key}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{model_key}] acc={res['accuracy']} none={res['none_accuracy']} "
          f"ooc={out_of_cand} avg={res['avg_latency_s']}s"
          + (f" cost=${res.get('cost_usd_total')}" if model_key == "gpt" else ""))


def summarize():
    rows = []
    for k in ("base", "gpt", "adapter", "gguf"):
        p = OUTDIR / f"{k}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    summary = [{kk: r.get(kk) for kk in
                ("model_key", "model", "n", "accuracy", "none_accuracy", "match_accuracy",
                 "out_of_candidates", "avg_latency_s", "cost_usd_total", "cost_usd_per_req")}
               for r in rows]
    (OUTDIR / "results.json").write_text(json.dumps(
        {"task": "sllm_compare", "commit": git_head(),
         "data": "keyword_selection/data/test_with_candidates.jsonl (389)",
         "prompt": "keyword_selection/prompt.py SYSTEM_PROMPT + build_user_prompt",
         "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt", "gguf", "base", "adapter"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run_model(a.model, a.limit)

#!/usr/bin/env python3
"""GSM8K against an OpenAI-compatible endpoint, on the checkpoint author's protocol.

Why this exists: the 17-scenario agentic tournament the README's defaults are chosen
by is not in this repo. But the RadixArk checkpoint publishes a GSM8K result with a
fully specified protocol (`gsm8k_metrics.json`, `qualification-notes.md`):

    full 1319, single-shot, sgl-eval @ 645cf56,
    temperature 0.6, top_p 0.95, max_tokens 8192, seed 0
    -> score 97.27% (1283/1319), stop_rate 98.86%, truncated 1.14%

That is a reproducible number on the same base model, so it is a usable quality gate
for a checkpoint swap -- the thing a speed benchmark cannot tell you. This reproduces
the protocol against any served arm so two of them can be compared on equal terms.

Note the published number came from SGLang, not vLLM, so it is a reference point and
not a like-for-like control. For a real A/B, run this against both arms yourself:
absolute scores may shift a little with the serving stack, but the *difference*
between two arms measured the same way is meaningful.

Answer extraction follows the usual GSM8K convention: gold is the number after '####',
prediction is the last number in the reply. With --reasoning-parser on the server, the
thinking trace arrives as `reasoning_content` and is deliberately NOT scanned unless
`content` came back empty -- scoring the scratchpad would inflate the result.

Usage:
  eval_gsm8k.py --base http://localhost:18300/v1 --model qwen3.8-flash-next
  eval_gsm8k.py --limit 200 --threads 8 --out ~/q38-tmp/gsm8k-ct.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The canonical test split, straight from the GSM8K repo: 1319 lines of
# {"question", "answer"} JSON. Plain JSONL on purpose -- the vLLM image has neither
# pandas nor pyarrow, and this tool should not need a pip install to run.
GSM8K_JSONL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
GSM8K_EXPECTED = 1319
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def die(msg: str) -> None:
    sys.exit(f"!! {msg}")


def load_gsm8k(cache: str) -> list[dict]:
    """The 1319-example test split, cached locally after the first fetch (~750 KB)."""
    path = os.path.join(cache, "gsm8k_test.jsonl")
    os.makedirs(cache, exist_ok=True)
    if not os.path.exists(path):
        with urllib.request.urlopen(GSM8K_JSONL, timeout=120) as r:
            body = r.read()
        with open(path, "wb") as f:
            f.write(body)
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if len(rows) != GSM8K_EXPECTED:
        die(f"expected {GSM8K_EXPECTED} examples, got {len(rows)} -- stale or partial cache at {path}")
    return rows


def gold_answer(answer: str) -> str:
    if "####" not in answer:
        die(f"malformed gold answer: {answer[:80]!r}")
    return answer.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> str | None:
    """Last number in the reply, commas stripped, trailing '.' tolerated."""
    nums = NUM_RE.findall(text or "")
    if not nums:
        return None
    v = nums[-1].replace(",", "").rstrip(".")
    return v or None


def numeric_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


class Runner:
    def __init__(self, base: str, model: str, gen: dict, timeout: int) -> None:
        self.base, self.model, self.gen, self.timeout = base.rstrip("/"), model, gen, timeout
        self.lock = threading.Lock()
        self.done = 0
        self.correct = 0

    def ask(self, question: str) -> tuple[str, str, str]:
        """Return (content, reasoning, finish_reason)."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            **self.gen,
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.load(r)
        ch = d["choices"][0]
        msg = ch.get("message") or {}
        return msg.get("content") or "", msg.get("reasoning_content") or "", ch.get("finish_reason") or ""

    def one(self, ex: dict, total: int) -> dict:
        gold = gold_answer(ex["answer"])
        rec = {"gold": gold, "pred": None, "ok": False, "finish": "", "error": None}
        try:
            content, reasoning, finish = self.ask(ex["question"])
            rec["finish"] = finish
            # Score the answer, not the scratchpad: only fall back to the thinking
            # trace when the model returned no visible content at all.
            pred = extract_pred(content) or (extract_pred(reasoning) if not content.strip() else None)
            rec["pred"] = pred
            rec["ok"] = pred is not None and numeric_eq(pred, gold)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self.done += 1
            self.correct += bool(rec["ok"])
            step = 5 if self.done <= 20 else 25
            if self.done % step == 0 or self.done == total:
                pct = 100.0 * self.done / total
                print(f"   {self.done}/{total} ({pct:.0f}%)  correct so far: {self.correct}",
                      flush=True)
        return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:18300/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--limit", type=int, default=0, help="0 = the full 1319")
    ap.add_argument("--threads", type=int, default=8, help="match the server's --max-num-seqs")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--cache", default=os.path.expanduser("~/q38-tmp/eval"))
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    data = load_gsm8k(args.cache)
    if args.limit:
        data = data[: args.limit]
    gen = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    print(f">> GSM8K: {len(data)} examples, {args.threads} threads, {json.dumps(gen)}", flush=True)
    print(f">> endpoint {args.base} model {args.model!r}", flush=True)
    print(">> reference: RadixArk published 97.27% (1283/1319) on this protocol", flush=True)

    runner = Runner(args.base, args.model, gen, args.timeout)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        recs = list(pool.map(lambda e: runner.one(e, len(data)), data))
    elapsed = time.time() - t0

    n = len(recs)
    errors = sum(1 for r in recs if r["error"])
    scored = n - errors
    correct = sum(1 for r in recs if r["ok"])
    stops = sum(1 for r in recs if r["finish"] == "stop")
    trunc = sum(1 for r in recs if r["finish"] == "length")
    result = {
        "name": "gsm8k",
        "label": args.label,
        "aggregate": {
            "score": correct / n if n else 0.0,
            "error_rate": errors / n if n else 0.0,
            "stop_rate": stops / n if n else 0.0,
            "truncated_rate": trunc / n if n else 0.0,
        },
        "num_examples": n,
        "num_correct": correct,
        "num_scored": scored,
        "gen": gen,
        "num_threads": args.threads,
        "latency_seconds": elapsed,
        "base_url": args.base,
        "model": args.model,
    }
    print(f"\n>> score {correct}/{n} = {result['aggregate']['score']*100:.2f}%")
    print(f"   stop {stops/n*100:.2f}%  truncated {trunc/n*100:.2f}%  errors {errors}")
    print(f"   {elapsed/60:.1f} min")
    print("   reference: RadixArk published 97.27% (1283/1319) on this protocol, via SGLang")
    if errors:
        first = next(r["error"] for r in recs if r["error"])
        print(f"   first error: {first}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({**result, "records": recs}, f, indent=2)
        print(f">> wrote {args.out}")


if __name__ == "__main__":
    main()

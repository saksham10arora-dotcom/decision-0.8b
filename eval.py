"""Score a decision model on data/test.jsonl.

  python eval.py                       # base model
  python eval.py --adapter adapters    # with a LoRA adapter

Metrics (overall and per task):
  acc        top option is the right one
  logloss    -log p(right option), lower is better
  brier      sum of squared errors over options, lower is better
  ece        calibration error: |confidence - accuracy| averaged over 10 confidence bins
  acc@80     accuracy on the decisions where top prob >= 0.8 (and how many there were)
  flip       % of decisions whose chosen option changes when the option order is shuffled
  mass       how much of the full-vocab probability landed on the option letters at all
"""
import argparse
import json
import math
import random
from pathlib import Path

from mlx_lm import load

from decision import decide

BASE = "Qwen/Qwen3.5-0.8B-Base"
HERE = Path(__file__).parent


def shuffled_order(n, rng):
    order = list(range(n))
    while order == list(range(n)):
        rng.shuffle(order)
    return order


def ece(confs, hits, bins=10):
    total, err = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            conf = sum(confs[i] for i in idx) / len(idx)
            acc = sum(hits[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return err


def summarize(rows):
    n = len(rows)
    confs = [max(r["probs"]) for r in rows]
    hits = [r["pred"] == r["answer"] for r in rows]
    sure = [h for c, h in zip(confs, hits) if c >= 0.8]
    return {
        "n": n,
        "acc": sum(hits) / n,
        "logloss": sum(-math.log(max(r["probs"][r["answer"]], 1e-12)) for r in rows) / n,
        "brier": sum(
            sum((p - (i == r["answer"])) ** 2 for i, p in enumerate(r["probs"])) for r in rows
        ) / n,
        "ece": ece(confs, hits),
        "acc@80": (sum(sure) / len(sure)) if sure else float("nan"),
        "n@80": len(sure),
        "flip": sum(r["flipped"] for r in rows) / n,
        "mass": sum(r["mass"] for r in rows) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load(BASE, adapter_path=args.adapter)
    recs = [json.loads(l) for l in open(HERE / "data" / f"{args.split}.jsonl")]
    rng = random.Random(1)

    rows = []
    for rec in recs:
        probs, mass = decide(model, tok, rec)
        pred = max(range(len(probs)), key=probs.__getitem__)
        order = shuffled_order(len(rec["options"]), rng)
        sprobs, _ = decide(model, tok, rec, order)
        spred = order[max(range(len(sprobs)), key=sprobs.__getitem__)]
        rows.append({**rec, "probs": probs, "pred": pred, "mass": mass, "flipped": spred != pred})

    report = {"all": summarize(rows)}
    for task in sorted({r["task"] for r in rows}):
        report[task] = summarize([r for r in rows if r["task"] == task])

    cols = ["n", "acc", "logloss", "brier", "ece", "acc@80", "n@80", "flip", "mass"]
    print(f"{'':10}" + "".join(f"{c:>9}" for c in cols))
    for name, m in report.items():
        print(f"{name:10}" + "".join(
            f"{m[c]:>9d}" if isinstance(m[c], int) else f"{m[c]:>9.3f}" for c in cols
        ))

    out = args.out or (HERE / "results" / f"{'base' if not args.adapter else Path(args.adapter).name}.json")
    Path(out).parent.mkdir(exist_ok=True)
    json.dump({"report": report, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

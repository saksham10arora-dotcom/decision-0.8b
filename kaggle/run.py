"""Decision model on Kaggle GPU: score the base model, LoRA-train all 24 layers, score again.

MODE = "smoke"  env check + 10 train steps, prints no metrics
MODE = "base"   score Qwen3.5-0.8B-Base on test
MODE = "train"  LoRA-train on train, score the tuned model on test

Same prompt and readout as decision.py (the MLX version): one forward pass,
softmax over the option-letter logits only. PyTorch's Qwen3.5 runs DeltaNet chunk by chunk,
so backprop stores one state per 64 tokens instead of one per token: all layers fit in 16GB.
"""
MODE = "train"

import glob
import json
import math
import os
import random
import subprocess
import sys
import time

if os.path.exists("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5.19", "peft>=0.21"], check=True)
    # the image ships torchao 0.10, which new peft rejects at import; we do not use it
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=True)

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-0.8B-Base"
LETTERS = "ABCD"
OUT = "/kaggle/working"
CFG = dict(rank=16, alpha=32, lr=2e-4, accum=8, epochs=1.0, warmup=20, seed=0)


def format_prompt(rec, order=None):
    opts = rec["options"]
    order = order if order is not None else list(range(len(opts)))
    lines = []
    if rec["situation"]:
        lines += ["Situation:", rec["situation"], ""]
    lines += [f"Question: {rec['question']}", "", "Options:"]
    lines += [f"{LETTERS[slot]}. {opts[i]}" for slot, i in enumerate(order)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def load_split(name):
    path = glob.glob(f"/kaggle/input/**/{name}.jsonl", recursive=True)[0]
    return [json.loads(l) for l in open(path)]


def shuffled_order(n, rng):
    order = list(range(n))
    while order == list(range(n)):
        rng.shuffle(order)
    return order


class Decider:
    def __init__(self, model, tok, dev):
        self.model, self.tok, self.dev = model, tok, dev
        self.ids = []
        for L in LETTERS:
            t = tok.encode(" " + L, add_special_tokens=False)
            assert len(t) == 1
            self.ids.append(t[0])

    def last_logits(self, rec, order=None):
        x = self.tok(format_prompt(rec, order), return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
        return self.model(input_ids=x, logits_to_keep=1).logits[0, -1].float()

    @torch.no_grad()
    def decide(self, rec, order=None):
        lp = self.last_logits(rec, order).log_softmax(-1)[self.ids[: len(rec["options"])]]
        return lp.softmax(-1).tolist(), lp.exp().sum().item()


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
        "brier": sum(sum((p - (i == r["answer"])) ** 2 for i, p in enumerate(r["probs"])) for r in rows) / n,
        "ece": ece(confs, hits),
        "acc@80": (sum(sure) / len(sure)) if sure else float("nan"),
        "n@80": len(sure),
        "flip": sum(r["flipped"] for r in rows) / n,
        "mass": sum(r["mass"] for r in rows) / n,
    }


def evaluate(decider, recs, name):
    decider.model.eval()
    rng = random.Random(1)
    rows = []
    for rec in recs:
        probs, mass = decider.decide(rec)
        pred = max(range(len(probs)), key=probs.__getitem__)
        order = shuffled_order(len(rec["options"]), rng)
        sprobs, _ = decider.decide(rec, order)
        spred = order[max(range(len(sprobs)), key=sprobs.__getitem__)]
        rows.append({**rec, "probs": probs, "pred": pred, "mass": mass, "flipped": spred != pred})
    report = {"all": summarize(rows)}
    for task in sorted({r["task"] for r in rows}):
        report[task] = summarize([r for r in rows if r["task"] == task])
    json.dump({"report": report, "rows": rows}, open(f"{OUT}/{name}.json", "w"), indent=1)
    return report


def print_report(report):
    cols = ["n", "acc", "logloss", "brier", "ece", "acc@80", "n@80", "flip", "mass"]
    print(f"{'':10}" + "".join(f"{c:>9}" for c in cols))
    for name, m in report.items():
        print(f"{name:10}" + "".join(f"{m[c]:>9d}" if isinstance(m[c], int) else f"{m[c]:>9.3f}" for c in cols))


def train(decider, recs, val, max_steps=None):
    model, ids = decider.model, decider.ids
    rng = random.Random(CFG["seed"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"], weight_decay=0.0)
    steps = max_steps or math.ceil(len(recs) * CFG["epochs"] / CFG["accum"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    )

    def example_loss(rec, r):
        order = list(range(len(rec["options"])))
        r.shuffle(order)
        target = torch.tensor(ids[order.index(rec["answer"])], device=decider.dev)
        return torch.nn.functional.cross_entropy(decider.last_logits(rec, order), target)

    def val_loss():
        model.eval()
        vr = random.Random(123)
        with torch.no_grad():
            v = sum(example_loss(r, vr).item() for r in val) / len(val)
        model.train()
        return v

    log = []
    if max_steps is None:
        log.append({"step": 0, "val_loss": val_loss()})
        print(log[-1], flush=True)
    model.train()
    queue, run, t0 = [], None, time.time()
    for step in range(1, steps + 1):
        tot = 0.0
        for _ in range(CFG["accum"]):
            if not queue:
                queue = list(range(len(recs)))
                rng.shuffle(queue)
            loss = example_loss(recs[queue.pop()], rng) / CFG["accum"]
            loss.backward()
            tot += loss.item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run = tot if run is None else 0.9 * run + 0.1 * tot
        if max_steps is not None:
            print(f"smoke step {step}  {time.time() - t0:.1f}s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB  finite={math.isfinite(tot)}", flush=True)
            continue
        if step % 25 == 0 or step == steps:
            print(f"step {step}/{steps}  train_loss {run:.3f}  {time.time() - t0:.0f}s", flush=True)
        if step % 125 == 0 or step == steps:
            log.append({"step": step, "train_loss": run, "val_loss": val_loss(), "secs": time.time() - t0})
            print(log[-1], flush=True)
    return log


def main():
    torch.manual_seed(CFG["seed"])
    dev = "cuda"
    print(torch.cuda.get_device_name(0), "bf16:", torch.cuda.is_bf16_supported(), "torch", torch.__version__, flush=True)
    # T4 says bf16 is "supported" but only emulates it; real bf16 needs compute capability 8+
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).to(dev)
    test, train_recs, val = load_split("test"), load_split("train"), load_split("val")
    print("data", len(train_recs), len(val), len(test), "dtype", dtype, flush=True)

    if MODE == "base":
        print_report(evaluate(Decider(model, tok, dev), test, "base"))
        return

    model = get_peft_model(model, LoraConfig(
        r=CFG["rank"], lora_alpha=CFG["alpha"], lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                        "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    decider = Decider(model, tok, dev)

    if MODE == "smoke":
        longest = sorted(train_recs, key=lambda r: -len(format_prompt(r)))[:CFG["accum"] * 2]
        train(decider, longest, val, max_steps=2)
        train(decider, train_recs, val, max_steps=8)
        decider.decide(test[0])
        print("SMOKE OK")
        return

    log = train(decider, train_recs, val)
    model.save_pretrained(f"{OUT}/adapter")
    json.dump({"cfg": CFG, "log": log}, open(f"{OUT}/train_log.json", "w"), indent=1)
    print_report(evaluate(decider, test, "tuned"))


if __name__ == "__main__":
    main()

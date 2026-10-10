"""Ask the decision model something. One forward pass per decision, no text generated.

  python decide.py "Is this email a phishing attempt?" --options Yes No \
      --situation "URGENT: verify your password within 24h at http://secure-paypa1.com"
  python decide.py --demo
"""
import argparse
import textwrap
import time

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision import LETTERS, format_prompt

BASE = "Qwen/Qwen3.5-0.8B-Base"

DEMO = [
    {
        "situation": "Subject: URGENT - your account will be suspended. Verify your password within 24 hours at http://secure-paypa1.com/login or lose access permanently.",
        "question": "Is this email a phishing attempt?",
        "options": ["Yes", "No"],
    },
    {
        "situation": "Nvidia shares jumped 8% in after-hours trading after the chipmaker raised its full-year revenue forecast on strong data center demand.",
        "question": "What is this article mainly about?",
        "options": ["World", "Sports", "Business", "Science/Technology"],
    },
    {
        "situation": "",
        "question": "Which of these is a renewable source of energy?",
        "options": ["coal", "natural gas", "wind", "oil"],
    },
]


def load(adapter="adapter"):
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter).to(dev).eval()
    ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    return model, tok, ids, dev


@torch.no_grad()
def decide(model, tok, ids, dev, rec):
    x = tok(format_prompt(rec), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    logits = model(input_ids=x, logits_to_keep=1).logits[0, -1].float()
    return logits[ids[: len(rec["options"])]].softmax(-1).tolist()


def show(rec, probs, ms):
    if rec["situation"]:
        print("\n" + textwrap.fill(rec["situation"], 76, initial_indent="  \033[2m", subsequent_indent="  ") + "\033[0m")
    print(f"\n  \033[1m{rec['question']}\033[0m\n")
    width = max(len(o) for o in rec["options"])
    best = max(range(len(probs)), key=probs.__getitem__)
    for i, (opt, p) in enumerate(zip(rec["options"], probs)):
        bar = "█" * round(p * 30) + "░" * (30 - round(p * 30))
        mark = "\033[1;32m" if i == best else "\033[2m"
        print(f"  {mark}{opt:<{width}}  {bar}  {p * 100:5.1f}%\033[0m")
    print(f"\n  \033[2m1 forward pass, {ms:.0f} ms\033[0m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--options", nargs="+")
    ap.add_argument("--situation", default="")
    ap.add_argument("--adapter", default="adapter")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if not args.demo and not (args.question and args.options and 2 <= len(args.options) <= 4):
        ap.error("give a question and 2-4 --options, or --demo")

    print("\n  \033[2mloading Qwen3.5-0.8B-Base + LoRA adapter...\033[0m", flush=True)
    model, tok, ids, dev = load(args.adapter)
    recs = DEMO if args.demo else [{"situation": args.situation, "question": args.question, "options": args.options}]
    decide(model, tok, ids, dev, recs[0])  # warm-up, so the timing below is the real per-decision cost
    for rec in recs:
        t0 = time.time()
        probs = decide(model, tok, ids, dev, rec)
        show(rec, probs, (time.time() - t0) * 1000)
    print()


if __name__ == "__main__":
    main()

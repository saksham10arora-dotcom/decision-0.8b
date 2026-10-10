"""LoRA fine-tune Qwen3.5-0.8B-Base into a decision model.

The objective: at the last prompt position ("Answer:"), put probability on the right letter.
Loss is cross-entropy over the full vocab on that one token, nothing else is trained.
Option order is reshuffled every time an example is seen, so position carries no signal.

  python train.py --out adapters
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.utils import linear_to_lora_layers

from decision import format_prompt, letter_ids

BASE = "Qwen/Qwen3.5-0.8B-Base"
HERE = Path(__file__).parent


def last_logits(model, toks, lengths, top):
    """Logits at each sequence's last real token only (full B x L x 248k vocab won't fit).

    Only the top `top` layers are trained. The ones below run frozen on the fast kernel,
    because backprop through Qwen3.5's linear-attention (DeltaNet) layers stores a
    16x128x128 state for every token, every layer: ~68GB at batch 8 for all 18 of them.
    """
    tm = model.language_model.model
    h = tm.embed_tokens(toks)
    fa_mask = create_attention_mask(h, None)
    for i, layer in enumerate(tm.layers):
        if i == len(tm.layers) - top:
            h = mx.stop_gradient(h)
        h = layer(h, mask=None if layer.is_linear else fa_mask)
    h = tm.norm(h)[mx.arange(h.shape[0]), lengths - 1]
    return tm.embed_tokens.as_linear(h)


def make_batch(recs, tok, ids, rng):
    seqs, targets = [], []
    for r in recs:
        order = list(range(len(r["options"])))
        rng.shuffle(order)
        seqs.append(tok.encode(format_prompt(r, order)))
        targets.append(ids[order.index(r["answer"])])
    L = max(len(s) for s in seqs)
    toks = mx.array([s + [0] * (L - len(s)) for s in seqs])
    return toks, mx.array([len(s) for s in seqs]), mx.array(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--top", type=int, default=1, help="train LoRA on the top N layers")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    model, tok = load(BASE)
    ids = letter_ids(tok, 4)
    lora_cfg = {"rank": args.rank, "scale": args.scale, "dropout": 0.0}
    model.freeze()
    linear_to_lora_layers(model, args.top, lora_cfg)
    # Qwen3.5's fast DeltaNet kernel has no backward pass: only the trained layers use the slow ops path
    model.eval()
    for layer in model.layers[-args.top:]:
        layer.train()
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.2f}M")

    train = [json.loads(l) for l in open(HERE / "data" / "train.jsonl")]
    val = [json.loads(l) for l in open(HERE / "data" / "val.jsonl")]

    def loss_fn(model, toks, lengths, targets):
        return nn.losses.cross_entropy(last_logits(model, toks, lengths, args.top), targets).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    steps = math.ceil(len(train) * args.epochs / args.batch)
    sched = optim.cosine_decay(args.lr, steps, end=args.lr * 0.1)
    opt = optim.Adam(learning_rate=sched)

    def val_loss():
        vrng = random.Random(123)
        tot = 0.0
        for i in range(0, len(val), args.batch):
            toks, lengths, targets = make_batch(val[i : i + args.batch], tok, ids, vrng)
            tot += loss_fn(model, toks, lengths, targets).item() * len(targets)
        return tot / len(val)

    print(f"step 0  val_loss {val_loss():.3f}")
    order, t0, run = [], time.time(), 0.0
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(train)))
            rng.shuffle(order)
        batch = [train[order.pop()] for _ in range(min(args.batch, len(order)))]
        toks, lengths, targets = make_batch(batch, tok, ids, rng)
        loss, grads = loss_and_grad(model, toks, lengths, targets)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        run = 0.9 * run + 0.1 * loss.item() if step > 1 else loss.item()
        if step % 25 == 0 or step == steps:
            print(f"step {step}/{steps}  train_loss {run:.3f}  {time.time() - t0:.0f}s", flush=True)
        if step % 125 == 0 or step == steps:
            print(f"step {step}  val_loss {val_loss():.3f}", flush=True)

    out = HERE / args.out
    out.mkdir(exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    json.dump(
        {"fine_tune_type": "lora", "num_layers": args.top, "lora_parameters": lora_cfg, **vars(args)},
        open(out / "adapter_config.json", "w"),
        indent=1,
    )
    print("saved", out)


if __name__ == "__main__":
    main()

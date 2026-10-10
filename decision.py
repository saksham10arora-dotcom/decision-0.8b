"""The decision model: one forward pass, read the option-letter logits, softmax over them.

Same prompt format for training and eval, so the only thing that changes is the weights.
"""
LETTERS = "ABCD"


def format_prompt(rec, order=None):
    """order[i] = which original option is shown in slot i. Default: original order."""
    opts = rec["options"]
    order = order if order is not None else list(range(len(opts)))
    lines = []
    if rec["situation"]:
        lines += ["Situation:", rec["situation"], ""]
    lines += [f"Question: {rec['question']}", "", "Options:"]
    lines += [f"{LETTERS[slot]}. {opts[i]}" for slot, i in enumerate(order)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def letter_ids(tokenizer, n):
    ids = []
    for L in LETTERS[:n]:
        t = tokenizer.encode(" " + L)
        assert len(t) == 1, f"' {L}' is {len(t)} tokens"
        ids.append(t[0])
    return ids


def decide(model, tokenizer, rec, order=None):
    """Returns (probs over shown slots, mass the full vocab put on the letters)."""
    import mlx.core as mx

    n = len(rec["options"])
    toks = mx.array(tokenizer.encode(format_prompt(rec, order)))[None]
    logits = model(toks)[0, -1].astype(mx.float32)
    logp_full = logits - mx.logsumexp(logits)
    ids = letter_ids(tokenizer, n)
    lp = logp_full[mx.array(ids)]
    mass = mx.exp(lp).sum()
    probs = mx.softmax(lp)
    mx.eval(probs, mass)
    return probs.tolist(), mass.item()

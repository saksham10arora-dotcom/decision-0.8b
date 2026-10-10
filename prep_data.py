"""Build decision-format data from three public sets.

Every record: situation (may be empty), question, options (list of str), answer (index).
Splits are made per source BEFORE anything else touches the data:
  train: 1000 per task (from the source's train split)
  val:    100 per task (from the source's train split, disjoint from train)
  test:   100 per task (from the source's validation/test split, never trained on)
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 0
N_TRAIN, N_VAL, N_TEST = 1000, 100, 100
OUT = Path(__file__).parent / "data"
MAX_SITUATION_CHARS = 1500


def boolq(row):
    return {
        "task": "yesno",
        "situation": row["passage"][:MAX_SITUATION_CHARS],
        "question": row["question"].strip().capitalize() + "?",
        "options": ["Yes", "No"],
        "answer": 0 if row["answer"] else 1,
    }


def arc(row):
    choices = row["choices"]["text"]
    labels = row["choices"]["label"]
    if len(choices) != 4 or row["answerKey"] not in labels:
        return None
    return {
        "task": "science",
        "situation": "",
        "question": row["question"],
        "options": choices,
        "answer": labels.index(row["answerKey"]),
    }


AG_LABELS = ["World", "Sports", "Business", "Science/Technology"]


def ag(row):
    return {
        "task": "news",
        "situation": row["text"][:MAX_SITUATION_CHARS],
        "question": "What is this article mainly about?",
        "options": AG_LABELS,
        "answer": row["label"],
    }


def take(rows, fn, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    out = []
    for r in rows:
        rec = fn(r)
        if rec is not None:
            out.append(rec)
        if len(out) == n:
            break
    assert len(out) == n, f"only got {len(out)}"
    return out


def main():
    rng = random.Random(SEED)
    sources = [
        ("google/boolq", None, "train", "validation", boolq),
        ("allenai/ai2_arc", "ARC-Easy", "train", "test", arc),
        ("fancyzhx/ag_news", None, "train", "test", ag),
    ]
    splits = {"train": [], "val": [], "test": []}
    for name, cfg, train_split, test_split, fn in sources:
        ds = load_dataset(name, cfg)
        pool = take(ds[train_split], fn, N_TRAIN + N_VAL, rng)
        splits["train"] += pool[:N_TRAIN]
        splits["val"] += pool[N_TRAIN:]
        splits["test"] += take(ds[test_split], fn, N_TEST, rng)

    OUT.mkdir(exist_ok=True)
    for split, recs in splits.items():
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        by_task = {}
        for r in recs:
            by_task[r["task"]] = by_task.get(r["task"], 0) + 1
        print(split, len(recs), by_task)


if __name__ == "__main__":
    main()

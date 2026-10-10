"""Draw the before/after dumbbell chart for the README, in light and dark.

Numbers come from the eval tables (kaggle/run.py, MODE base and train). Run: python assets/make_charts.py
"""
from pathlib import Path

ROWS = [
    # label, note, base, tuned
    ("Accuracy", "300 held-out questions, higher is better", 81.0, 85.3),
    ("Yes/no accuracy, “No” listed first", "higher is better", 62.0, 82.0),
    ("Probability on A/B/C/D at all", "the rest goes to other words, higher is better", 44.0, 99.9),
    ("Answers that flip when reordered", "all 300 questions, lower is better", 23.0, 8.7),
    ("Yes/no answers that flip when reordered", "lower is better", 46.0, 14.0),
]

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e8e7e3",
                  link="#c9c8c3", base="#86b6ef", tuned="#1c5cab"),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", grid="#2e2e2b",
                 link="#4a4a46", base="#184f95", tuned="#6da7ec"),
}

W, X0, X1 = 900, 410, 810
TOP, ROW_H = 132, 62
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def x(v):
    return X0 + (X1 - X0) * v / 100


def svg(t):
    H = TOP + ROW_H * len(ROWS) + 48
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
        f'<rect width="{W}" height="{H}" rx="14" fill="{t["surface"]}"/>',
        f'<text x="32" y="44" font-size="21" font-weight="650" fill="{t["text"]}">Fine-tuning fixed the order bias more than the accuracy</text>',
        f'<text x="32" y="70" font-size="14" fill="{t["muted"]}">Qwen3.5-0.8B-Base before and after one LoRA epoch (3,000 examples, 14 min on a free Kaggle T4)</text>',
    ]
    # legend
    for i, (name, key) in enumerate([("Base model", "base"), ("Fine-tuned", "tuned")]):
        lx = 32 + i * 150
        o.append(f'<circle cx="{lx + 7}" cy="99" r="7" fill="{t[key]}"/>')
        o.append(f'<text x="{lx + 22}" y="104" font-size="14" fill="{t["text"]}">{name}</text>')
    # grid + ticks
    gy0, gy1 = TOP - 18, TOP + ROW_H * len(ROWS) - 18
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{x(v)}" y1="{gy0}" x2="{x(v)}" y2="{gy1}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x(v)}" y="{gy1 + 22}" font-size="12" fill="{t["muted"]}" text-anchor="middle" style="font-variant-numeric: tabular-nums">{v}%</text>')
    for i, (label, note, b, a) in enumerate(ROWS):
        cy = TOP + ROW_H * i + 12
        o.append(f'<text x="32" y="{cy - 2}" font-size="15" font-weight="550" fill="{t["text"]}">{label}</text>')
        o.append(f'<text x="32" y="{cy + 17}" font-size="12" fill="{t["muted"]}">{note}</text>')
        o.append(f'<line x1="{x(b)}" y1="{cy}" x2="{x(a)}" y2="{cy}" stroke="{t["link"]}" stroke-width="2"/>')
        up = a > b
        for v, key, anchor_out in ((b, "base", not up), (a, "tuned", up)):
            o.append(f'<circle cx="{x(v)}" cy="{cy}" r="7" fill="{t[key]}" stroke="{t["surface"]}" stroke-width="2"/>')
            dx, anchor = (14, "start") if anchor_out else (-14, "end")
            weight, fill = ("650", t["text"]) if key == "tuned" else ("400", t["muted"])
            val = f"{v:.1f}".rstrip("0").rstrip(".")
            o.append(f'<text x="{x(v) + dx}" y="{cy + 5}" font-size="14" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{val}%</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    here = Path(__file__).parent
    for name, t in THEMES.items():
        (here / f"results-{name}.svg").write_text(svg(t))
        print("wrote", here / f"results-{name}.svg")

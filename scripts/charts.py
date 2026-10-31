"""Render the README charts from results/metrics and results/bench.

Static PNGs for GitHub, so there is no hover layer: every bar carries its value as a
direct label instead, and the numbers are also in the README tables. Colors follow the
fixed categorical order of the reference palette (blue, orange, aqua); text uses ink
colors, never series colors; reference lines are muted gray.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "docs" / "charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#9b9a95"


def metric(system: str, split: str) -> dict | None:
    p = METRICS / f"{system}__{split}.json"
    return json.loads(p.read_text()) if p.exists() else None


def style(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="semibold", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)


def figure(width: float = 7.2, height: float = 3.8) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def ci(m: dict, key: str = "accuracy") -> tuple[float, float, float]:
    v = m[key]["value"]
    lo, hi = m[key]["ci95"]
    return v, v - lo, hi - v


def reference_line(ax: plt.Axes, value: float, label: str, x: float) -> None:
    ax.axhline(value, color=REFERENCE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x, value + 0.008, label, color=INK_2, fontsize=8.5, va="bottom")


def learning_curve() -> None:
    points = [(500, "lora-n500"), (1500, "lora-n1500"), (3000, "lora-n3000"), (6353, "lora-main")]
    rows = [(n, metric(s, "test")) for n, s in points]
    rows = [(n, m) for n, m in rows if m]
    if len(rows) < 2:
        return
    fig, ax = figure()
    xs = [n for n, _ in rows]
    vals = [ci(m) for _, m in rows]
    ys = [v for v, _, _ in vals]
    lo = [v - a for v, a, _ in vals]
    hi = [v + b for v, _, b in vals]
    ax.fill_between(xs, lo, hi, color=SERIES[0], alpha=0.12, linewidth=0)
    ax.plot(
        xs,
        ys,
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=7,
        zorder=3,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y + 0.012, f"{y:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    if base := metric("base", "test"):
        reference_line(ax, base["accuracy"]["value"], "base zero-shot", xs[0])
    if fs := metric("base-fewshot", "test"):
        reference_line(ax, fs["accuracy"]["value"], "base few-shot", xs[0])
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("training examples (one epoch, LoRA rank 16)", color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(
        ax,
        "Learning curve",
        "Exact-match accuracy on the 499-example held-out test split (unseen tools), 95% bootstrap CI",
    )
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", facecolor=SURFACE)
    plt.close(fig)


def bars(name: str, title: str, subtitle: str, items: list[tuple[str, str]], split: str) -> None:
    rows = [(label, metric(s, split)) for label, s in items]
    rows = [(label, m) for label, m in rows if m]
    if len(rows) < 2:
        return
    fig, ax = figure(height=3.6)
    labels = [label for label, _ in rows]
    vals = [ci(m) for _, m in rows]
    ys = [v for v, _, _ in vals]
    err = [[a for _, a, _ in vals], [b for _, _, b in vals]]
    ax.bar(labels, ys, width=0.55, color=SERIES[0], zorder=2)
    ax.errorbar(labels, ys, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=3)
    for i, y in enumerate(ys):
        ax.text(
            i, y + err[1][i] + 0.012, f"{y:.1%}", ha="center", va="bottom", fontsize=9, color=INK
        )
    top = max(y + e for y, e in zip(ys, err[1], strict=True))
    ax.set_ylim(0, min(1.0, top + 0.12))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", facecolor=SURFACE)
    plt.close(fig)


def quantization() -> None:
    """Accuracy of the same fine-tune at three precisions, one panel per eval split."""
    splits = [("test", "held-out test"), ("handchecked", "hand-checked"), ("bfcl", "BFCL subset")]
    systems = [("bf16", "ft"), ("8-bit", "ft-8bit"), ("4-bit", "ft-4bit")]
    data = {sp: [metric(s, sp) for _, s in systems] for sp, _ in splits}
    if not all(all(row) for row in data.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (sp, label) in zip(axes, splits, strict=True):
        vals = [ci(m) for m in data[sp]]  # type: ignore[arg-type]
        ys = [v for v, _, _ in vals]
        err = [[a for _, a, _ in vals], [b for _, _, b in vals]]
        names = [n for n, _ in systems]
        ax.bar(names, ys, width=0.6, color=SERIES[0], zorder=2)
        ax.errorbar(
            names, ys, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=3
        )
        for i, y in enumerate(ys):
            ax.text(i, y + err[1][i] + 0.02, f"{y:.1%}", ha="center", fontsize=8.5, color=INK)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(label, fontsize=10, color=INK_2, loc="left")
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle(
        "Quantizing the fine-tune", x=0.01, ha="left", fontsize=12, color=INK, fontweight="semibold"
    )
    fig.text(
        0.01,
        0.885,
        "Same LoRA fine-tune, fused, then converted with mlx_lm. Exact-match accuracy, 95% bootstrap CI",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "quantization.png", facecolor=SURFACE)
    plt.close(fig)


def loss_curve() -> None:
    path = ROOT / "runs" / "main" / "loss.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    train = [(r["iteration"], r["train_loss"]) for r in rows if r["kind"] == "train"]
    val = [(r["iteration"], r["val_loss"]) for r in rows if r["kind"] == "val"]
    fig, ax = figure(height=3.4)
    ax.plot(*zip(*train, strict=True), color=SERIES[0], linewidth=1.2, alpha=0.8, label="train")
    ax.plot(
        *zip(*val, strict=True),
        color=SERIES[1],
        linewidth=2,
        marker="o",
        markersize=4,
        label="validation",
    )
    ax.set_yscale("log")
    ax.set_xlabel("iteration (batch 2, gradient accumulation 2)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    style(
        ax,
        "Training loss, main run",
        "Masked cross-entropy on the assistant turn only; validation every 100 iterations (log scale)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loss_curve()
    learning_curve()
    bars(
        "lora_rank",
        "LoRA rank",
        "1,500 training examples, everything else fixed. Held-out test accuracy, 95% bootstrap CI",
        [("rank 8", "lora-n1500-r8"), ("rank 16", "lora-n1500"), ("rank 64", "lora-n1500-r64")],
        "test",
    )
    quantization()
    print("charts written to", OUT)


if __name__ == "__main__":
    main()

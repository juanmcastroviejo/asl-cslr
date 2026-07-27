"""Aggregate finished runs into the tables and figures used in the paper.

    python -m src.evaluate --runs results/full_bilstm results/hands_bilstm ...

Writes to results/figures/ and results/tables/:
    figure_wer_by_length.png     RQ1  -- degradation with phrase length
    figure_ablation.png          RQ2  -- contribution of pose / face landmarks
    figure_latency.png           RQ3  -- accuracy/latency trade-off
    figure_confusion_<run>.png        -- most-confused sign pairs
    figure_training_curves.png        -- loss and validation WER
    table_main_results.csv/.md        -- headline comparison table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import confusion_matrix, confusion_pairs, wer_by_length


def load_run(d: Path) -> dict | None:
    f = d / "summary.json"
    if not f.exists():
        print(f"  (skipping {d}: no summary.json)")
        return None
    s = json.loads(f.read_text())
    s["dir"] = d
    for k in ("refs", "hyps"):
        p = d / f"test_{k}.npy"
        s[k] = [list(x) for x in np.load(p, allow_pickle=True)] if p.exists() else []
    return s


def fig_wer_by_length(runs, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for r in runs:
        if not r["refs"]:
            continue
        by = wer_by_length(r["refs"], r["hyps"])
        xs = sorted(by)
        plt.plot(xs, [by[x]["wer"] for x in xs], marker="o", label=r["run"])
    plt.xlabel("Phrase length (number of glosses)")
    plt.ylabel("Word error rate")
    plt.title("WER as a function of phrase length")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "figure_wer_by_length.png", dpi=200)
    plt.close()


def fig_ablation(runs, out: Path) -> None:
    order = ["hands", "hands_pose", "full"]
    labels = {"hands": "Hands only\n(126-d)", "hands_pose": "Hands + pose\n(258-d)",
              "full": "Holistic\n(1662-d)"}
    heads = sorted({r["head"] for r in runs})
    width = 0.8 / max(len(heads), 1)
    plt.figure(figsize=(7, 4.5))
    for i, head in enumerate(heads):
        xs, ys = [], []
        for j, g in enumerate(order):
            m = [r for r in runs if r["group"] == g and r["head"] == head]
            if m:
                xs.append(j + i * width)
                ys.append(m[0]["test_beam"]["wer"])
        plt.bar(xs, ys, width, label=head)
        for x, y in zip(xs, ys):
            plt.text(x, y + 0.008, f"{y:.3f}", ha="center", fontsize=8)
    plt.xticks([j + width * (len(heads) - 1) / 2 for j in range(len(order))],
               [labels[g] for g in order])
    plt.ylabel("Test WER (beam search, width 10)")
    plt.title("Feature ablation: contribution of pose and facial landmarks")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "figure_ablation.png", dpi=200)
    plt.close()


def fig_latency(runs, latency: dict | None, out: Path) -> None:
    if not latency:
        return
    plt.figure(figsize=(7, 4.5))
    for r in runs:
        lat = latency.get(r["run"])
        if not lat:
            continue
        plt.scatter(lat["model_ms_mean"], r["test_beam"]["wer"], s=90)
        plt.annotate(r["run"], (lat["model_ms_mean"], r["test_beam"]["wer"]),
                     textcoords="offset points", xytext=(6, 5), fontsize=8)
    plt.axvline(80, ls="--", c="r", alpha=0.6)
    plt.text(80, plt.ylim()[1] * 0.95, " 80 ms target", color="r", fontsize=8)
    plt.xlabel("Mean model inference latency per window (ms, CPU)")
    plt.ylabel("Test WER")
    plt.title("Accuracy / latency trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "figure_latency.png", dpi=200)
    plt.close()


def fig_confusion(r, out: Path) -> None:
    if not r["refs"]:
        return
    vocab = r["vocab"]
    M = confusion_matrix(r["refs"], r["hyps"], len(vocab))
    norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(max(6, len(vocab) * 0.55), max(5, len(vocab) * 0.5)))
    plt.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Proportion of reference tokens")
    ticks = list(vocab) + ["<del/ins>"]
    plt.xticks(range(len(ticks)), ticks, rotation=90, fontsize=7)
    plt.yticks(range(len(ticks)), ticks, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.title(f"Confusion matrix ({r['run']})")
    plt.tight_layout()
    plt.savefig(out / f"figure_confusion_{r['run']}.png", dpi=200)
    plt.close()


def fig_curves(runs, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in runs:
        h = r.get("history", [])
        if not h:
            continue
        ep = [x["epoch"] for x in h]
        ax[0].plot(ep, [x["loss"] for x in h], label=r["run"])
        ax[1].plot(ep, [x["val_wer"] for x in h], label=r["run"])
    ax[0].set(xlabel="Epoch", ylabel="CTC loss", title="Training loss")
    ax[1].set(xlabel="Epoch", ylabel="Validation WER", title="Validation WER")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "figure_training_curves.png", dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--latency", default="results/latency.json")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    runs = [r for r in (load_run(Path(d)) for d in a.runs) if r]
    if not runs:
        raise SystemExit("No usable runs found.")
    figs = Path(a.out) / "figures"
    tabs = Path(a.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    lat = json.loads(Path(a.latency).read_text()) if Path(a.latency).exists() else None

    fig_wer_by_length(runs, figs)
    fig_ablation(runs, figs)
    fig_latency(runs, lat, figs)
    fig_curves(runs, figs)
    for r in runs:
        fig_confusion(r, figs)

    # ---- main results table ------------------------------------------
    hdr = ["run", "features", "head", "params", "val_WER", "WER_greedy",
           "WER_beam", "sub", "del", "ins", "sent_acc", "latency_ms"]
    rows = []
    for r in sorted(runs, key=lambda x: x["test_beam"]["wer"]):
        L = (lat or {}).get(r["run"], {})
        b = r["test_beam"]
        rows.append([r["run"], r["group"], r["head"], f"{r['params']:,}",
                     f"{r['best_val_wer']:.4f}", f"{r['test_greedy']['wer']:.4f}",
                     f"{b['wer']:.4f}", f"{b['sub']:.4f}", f"{b['del']:.4f}",
                     f"{b['ins']:.4f}", f"{b['sentence_acc']:.4f}",
                     f"{L.get('model_ms_mean', float('nan')):.1f}" if L else "n/a"])
    (tabs / "table_main_results.csv").write_text(
        "\n".join([",".join(hdr)] + [",".join(r) for r in rows]))
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (tabs / "table_main_results.md").write_text("\n".join(md))

    # ---- per-length table --------------------------------------------
    lines = ["| run | length | n_phrases | WER | sentence acc |", "|---|---|---|---|---|"]
    for r in runs:
        if not r["refs"]:
            continue
        for L, m in wer_by_length(r["refs"], r["hyps"]).items():
            lines.append(f"| {r['run']} | {L} | {m['n_phrases']} | "
                         f"{m['wer']:.4f} | {m['sentence_acc']:.4f} |")
    (tabs / "table_wer_by_length.md").write_text("\n".join(lines))

    # ---- top confusions ----------------------------------------------
    lines = ["| run | reference | predicted | count |", "|---|---|---|---|"]
    for r in runs:
        if not r["refs"]:
            continue
        for (a_, b_), c in confusion_pairs(r["refs"], r["hyps"]).most_common(10):
            lines.append(f"| {r['run']} | {r['vocab'][a_ - 1]} | {r['vocab'][b_ - 1]} | {c} |")
    (tabs / "table_confusions.md").write_text("\n".join(lines))

    print("\n".join(md))
    print(f"\nFigures -> {figs}\nTables  -> {tabs}")


if __name__ == "__main__":
    main()

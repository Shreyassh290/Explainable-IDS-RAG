"""Generate updated V2 evaluation figures from final D5 gated results.

Reads ONLY from artifacts/v2b/eval/eval_negfix_gated.json (source of truth).
Writes new PNGs to artifacts/v2/plots/:
  - v2_mixed_confusion_gated.png  (row-normalised 6x6 confusion matrix)
  - v2_mixed_recall_gated.png     (per-class recall bar chart)

Old v2_mixed_confusion.png and v2_mixed_recall.png are left untouched
(historical / pre-gating, still referenced with that label in the UI).

Run with:
  uv run python scripts/make_v2_gated_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES_ALL_6 = ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]


def _load_gated_json(repo_root: Path) -> dict:
    p = repo_root / "artifacts" / "v2b" / "eval" / "eval_negfix_gated.json"
    if not p.is_file():
        raise FileNotFoundError(f"Source of truth not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_gated_confusion(summary: dict, out_path: Path) -> Path:
    """Row-normalised 6x6 confusion matrix from D5 gated (with FTP) results."""
    mixed = summary["gated_with_ftp"]["mixed_sample_15008"]
    classes = mixed["classes"]
    cm = np.array(mixed["confusion_matrix"], dtype=float)
    counts = np.array(mixed["confusion_matrix"], dtype=int)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    title = "Six-Class Gated Decision Rule — Mixed Test Sample (15,008 flows)"

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row % (Recall)")

    n_cls = len(classes)
    ax.set(
        xticks=np.arange(n_cls),
        yticks=np.arange(n_cls),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = 50.0
    for i in range(n_cls):
        for j in range(n_cls):
            pct = cm_norm[i, j]
            cnt = counts[i, j]
            text = f"{pct:.1f}%\n({cnt:,})"
            color = "white" if pct > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gated_recall(summary: dict, out_path: Path) -> Path:
    """Per-class recall bar chart from D5 gated (with FTP) results."""
    mixed = summary["gated_with_ftp"]["mixed_sample_15008"]
    per_class = mixed["per_class_metrics"]

    classes = [c for c in CLASSES_ALL_6 if c in per_class]
    recalls = [float(per_class[c]["recall"]) * 100.0 for c in classes]
    indicative = [per_class[c].get("is_indicative_only", False) for c in classes]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    title = "Per-Class Recall — Six-Class Gated Decision Rule (15,008-flow mixed sample)"

    x_pos = np.arange(len(classes))
    colors = ["#e15759" if ind else "#4e79a7" for ind in indicative]
    bars = ax.bar(x_pos, recalls, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    labels = []
    for c, bar, ind in zip(classes, bars, indicative):
        if ind:
            bar.set_hatch("//")
            labels.append(f"{c}\n(indicative only,\n8 test flows)")
        else:
            labels.append(c)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_ylim([0, 110])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, rec, ind in zip(bars, recalls, indicative):
        height = bar.get_height()
        label_text = f"{rec:.2f}%" + (" *" if ind else "")
        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.text(
        0.99, 0.02,
        "* FTP-BruteForce: indicative only (8 test flows)\nSource: D5 gated, artifacts/v2b/eval/eval_negfix_gated.json",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7.5, color="gray", style="italic",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(repo_root: Path | None = None) -> None:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    data = _load_gated_json(repo_root)
    plots_dir = repo_root / "artifacts" / "v2" / "plots"

    p_cm = plots_dir / "v2_mixed_confusion_gated.png"
    plot_gated_confusion(data, p_cm)
    print(f"Generated: {p_cm}")

    p_rec = plots_dir / "v2_mixed_recall_gated.png"
    plot_gated_recall(data, p_rec)
    print(f"Generated: {p_rec}")

    print("Done. Old v2_mixed_confusion.png and v2_mixed_recall.png preserved as historical.")


if __name__ == "__main__":
    main()


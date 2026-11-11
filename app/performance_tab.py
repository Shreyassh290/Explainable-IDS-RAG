"""Model Performance tab for the Streamlit dashboard.

Shows the saved evaluation graphs (ROC curves, confusion matrix, training
curves) and headline numbers read from artifacts/metrics.json. Nothing is
recomputed here: every figure comes from files written by the evaluation and
training steps, and a missing file produces a note instead of an error.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

CLASSES = ("Benign", "DoS", "DDoS")

BENIGN_CURVE_NOTE = (
    " Regenerated on 10 Nov 2026 after a retraining run; the model file is "
    "byte-identical to the one used for the evaluation."
)


def load_headline_metrics(metrics_path: Path) -> dict[str, Any] | None:
    """Read the headline numbers from metrics.json, or return None if unusable."""
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        report = data["classification_report"]
        return {
            "accuracy": float(report["accuracy"]),
            "flows": int(report["macro avg"]["support"]),
            "f1": {c: float(report[c]["f1-score"]) for c in CLASSES},
            "benign_fpr": float(data["benign_fpr_analysis"]["false_positive_rate"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _item(artifacts_dir: Path, name: str, caption: str) -> dict[str, Any]:
    path = artifacts_dir / name
    exists = path.is_file()
    generated = None
    if exists:
        generated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {
        "name": name,
        "path": path,
        "caption": caption,
        "exists": exists,
        "generated": generated,
    }


def collect_performance_assets(repo_root: Path) -> dict[str, Any]:
    """Describe every graph and the headline metrics; no Streamlit needed."""
    artifacts_dir = Path(repo_root) / "artifacts"
    curves = []
    for cls in CLASSES:
        caption = f"{cls} classifier: training and validation curves."
        if cls == "Benign":
            caption += BENIGN_CURVE_NOTE
        curves.append(_item(artifacts_dir, f"training_curves_{cls}.png", caption))
    return {
        "roc": _item(
            artifacts_dir,
            "roc_curves.png",
            "ROC curves for the three classifiers on the test set.",
        ),
        "confusion": _item(
            artifacts_dir,
            "confusion_matrix.png",
            "Confusion matrix on the test set.",
        ),
        "curves": curves,
        "metrics": load_headline_metrics(artifacts_dir / "metrics.json"),
    }


def _show(st: Any, item: dict[str, Any]) -> None:
    if item["exists"]:
        st.image(
            str(item["path"]),
            caption=f"{item['caption']} (file generated {item['generated']})",
        )
    else:
        st.info(f"{item['name']} not found in artifacts/.")


def render_performance_tab(repo_root: Path) -> None:
    import streamlit as st

    assets = collect_performance_assets(repo_root)
    m = assets["metrics"]

    st.subheader("Model Performance (CICIDS2018 test set)")
    if m is None:
        st.warning("artifacts/metrics.json is missing or unreadable; headline numbers are unavailable.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{m['accuracy']:.4f}")
        c2.metric("Test flows", f"{m['flows']:,}")
        c3.metric("Benign false-positive rate", f"{100 * m['benign_fpr']:.2f}%")
        st.caption("F1 per class: " + ", ".join(f"{c} {m['f1'][c]:.4f}" for c in CLASSES))

    left, right = st.columns(2)
    with left:
        _show(st, assets["roc"])
    with right:
        _show(st, assets["confusion"])

    st.subheader("Training curves")
    cols = st.columns(3)
    for col, item in zip(cols, assets["curves"]):
        with col:
            _show(st, item)

    with st.expander("How this compares with the reference paper"):
        ours = f"{100 * m['accuracy']:.2f}%" if m else "n/a"
        st.markdown(
            "| | Paper | This project |\n"
            "|---|---|---|\n"
            f"| CICIDS2018 accuracy | 99.84% | {ours} |\n"
            "| UNSW-NB15 accuracy | 95.30% | not done |\n"
            "| Features | 66 | 41 |\n"
            "| Classifier layout | 128-64-32, ReLU | 128-64-32, ReLU, dropout 0.3 |\n"
        )
        st.caption(
            "The test split is this project's own (the paper does not describe its split), "
            "so the match shows that the detection step works, not that the two setups are identical."
        )

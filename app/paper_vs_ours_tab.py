"""Paper vs Ours comparison tab for the Streamlit dashboard.

Compares benchmark numbers from the reference publication with the metrics
produced by our pipeline runs. Dynamic numbers are read directly from saved
evaluation artifacts; missing artifacts display 'not computed' without errors.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def resolve_paper_image(paper_dir: Path, filename: str) -> Path | None:
    """Resolve a paper graphic from app/assets/paper handling potential double extensions."""
    candidates = [
        paper_dir / filename,
        paper_dir / f"{filename}.png",
        paper_dir / f"{filename}.png.png",
    ]
    if filename.endswith(".png"):
        stem = filename[:-4]
        candidates.append(paper_dir / f"{stem}.png.png")

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_retrieval_metrics(retrieval_json: Path) -> tuple[str, str]:
    """Extract MRR and success rate from retrieval_eval.json if present."""
    if not retrieval_json.is_file():
        return "not computed", "not computed"
    try:
        with open(retrieval_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", [])
        if not rows:
            return "not computed", "not computed"

        rr_vals = [float(r["rr"]) for r in rows if "rr" in r]
        succ_vals = [int(r["success"]) for r in rows if "success" in r]

        mrr_str = f"{sum(rr_vals) / len(rr_vals):.2f}" if rr_vals else "not computed"
        if succ_vals:
            rate = sum(succ_vals) / len(succ_vals)
            succ_str = f"{rate:.2f} ({sum(succ_vals)}/{len(succ_vals)})"
        else:
            succ_str = "not computed"
        return mrr_str, succ_str
    except Exception:
        return "not computed", "not computed"


def load_citation_rate(eval_dir: Path) -> str:
    """Compute average citations per explanation from citation check artifacts if present."""
    csv_file = eval_dir / "citation_check" / "citation_check_per_report.csv"
    if csv_file.is_file():
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                c_vals = [int(r["citations"]) for r in reader if "citations" in r and r["citations"].isdigit()]
            if c_vals:
                return f"{sum(c_vals) / len(c_vals):.1f}"
        except Exception:
            pass

    summary_file = eval_dir / "citation_check" / "citation_check_summary.txt"
    if summary_file.is_file():
        try:
            text = summary_file.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "unique citation rows:" in line and "in flow_summary.csv: 30" in text:
                    # e.g. unique citation rows: 106 across 30 reports = 3.5
                    parts = line.split("unique citation rows:")
                    if len(parts) > 1:
                        num_str = parts[1].split(";")[0].strip()
                        if num_str.isdigit():
                            return f"{int(num_str) / 30.0:.1f}"
        except Exception:
            pass

    return "not computed"


def collect_paper_vs_ours_metrics(repo_root: Path) -> list[dict[str, str]]:
    """Gather comparison table rows, loading pipeline numbers dynamically from disk."""
    artifacts_dir = Path(repo_root) / "artifacts"
    metrics_json = artifacts_dir / "metrics.json"
    eval_dir = artifacts_dir / "eval"
    retrieval_json = eval_dir / "retrieval_eval.json"

    # Dynamic pipeline numbers default to 'not computed'
    ours_acc = "not computed"
    ours_roc_benign = "not computed"
    ours_roc_dos = "not computed"
    ours_roc_ddos = "not computed"

    if metrics_json.is_file():
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls_rep = data.get("classification_report", {})
            if "accuracy" in cls_rep:
                ours_acc = f"{float(cls_rep['accuracy']) * 100:.2f}%"

            rocs = data.get("roc_aucs", {})
            if "Benign" in rocs:
                ours_roc_benign = f"{float(rocs['Benign']):.4f}"
            if "DoS" in rocs:
                ours_roc_dos = f"{float(rocs['DoS']):.4f}"
            if "DDoS" in rocs:
                ours_roc_ddos = f"{float(rocs['DDoS']):.4f}"
        except Exception:
            pass

    ours_mrr, ours_succ = load_retrieval_metrics(retrieval_json)
    ours_cit = load_citation_rate(eval_dir)

    rows = [
        {
            "Metric": "CICIDS2018 test accuracy",
            "Paper": "99.84%",
            "Ours": ours_acc,
            "Note": "Section: Conclusion, text",
        },
        {
            "Metric": "ROC AUC (Benign)",
            "Paper": "0.9999",
            "Ours": ours_roc_benign,
            "Note": "Fig. 2 legend",
        },
        {
            "Metric": "ROC AUC (DoS)",
            "Paper": "0.9996",
            "Ours": ours_roc_dos,
            "Note": "Fig. 2 legend",
        },
        {
            "Metric": "ROC AUC (DDoS)",
            "Paper": "1.0000",
            "Ours": ours_roc_ddos,
            "Note": "Fig. 2 legend",
        },
        {
            "Metric": "Retrieval MRR",
            "Paper": "0.87",
            "Ours": ours_mrr,
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval success rate",
            "Paper": "0.97 (32/33)",
            "Ours": ours_succ,
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval Precision@5",
            "Paper": "0.91",
            "Ours": "not computed",
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval Recall@5",
            "Paper": "0.84",
            "Ours": "not computed",
            "Note": "Table VIII",
        },
        {
            "Metric": "Citations per explanation",
            "Paper": "4.2",
            "Ours": ours_cit,
            "Note": "Table VIII",
        },
    ]
    return rows


def collect_v2_extension_data(repo_root: Path) -> dict[str, Any] | None:
    """Read extra classes evaluation summary from artifacts/v2/eval/eval_summary_v2.json."""
    summary_path = Path(repo_root) / "artifacts" / "v2" / "eval" / "eval_summary_v2.json"
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_paper_vs_ours_tab(repo_root: Path) -> None:
    """Render the Paper vs Ours comparison dashboard."""
    st.header("Comparative Analysis: Reference Paper vs Our Pipeline")
    st.caption("Benchmark fidelity review across neural detection, retrieval metrics, and training trajectories.")

    # 1. Headline Comparison Table
    st.subheader("1. Headline Benchmark Comparison")
    table_data = collect_paper_vs_ours_metrics(repo_root)
    df_comp = pd.DataFrame(table_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Side-by-Side Visual Comparison
    st.subheader("2. Side-by-Side Visual Comparison")
    paper_dir = Path(repo_root) / "app" / "assets" / "paper"
    artifacts_dir = Path(repo_root) / "artifacts"

    # Row 1: Fig. 2 ROC Curves
    st.markdown("#### Row 1: ROC Curves (CICIDS2018)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reference Paper (Fig. 2)**")
        img_p1 = resolve_paper_image(paper_dir, "fig2_roc_cicids.png")
        if img_p1:
            st.image(str(img_p1), use_container_width=True)
        else:
            st.info("Drop fig2_roc_cicids.png into app/assets/paper/ to view side-by-side.")
    with col2:
        st.markdown("**Our Pipeline (artifacts/roc_curves.png)**")
        img_o1 = artifacts_dir / "roc_curves.png"
        if img_o1.is_file():
            st.image(str(img_o1), use_container_width=True)
        else:
            st.info("Missing artifacts/roc_curves.png")
    st.caption("Compare curve shape and final values only.")

    st.markdown("---")

    # Row 2: Fig. 4 Validation Accuracy
    st.markdown("#### Row 2: Validation Accuracy per Epoch")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Reference Paper (Fig. 4)**")
        img_p2 = resolve_paper_image(paper_dir, "fig4_val_accuracy.png")
        if img_p2:
            st.image(str(img_p2), use_container_width=True)
        else:
            st.info("Drop fig4_val_accuracy.png into app/assets/paper/ to view side-by-side.")
    with col4:
        st.markdown("**Our Pipeline (artifacts/training_curves_DoS.png)**")
        img_o2 = artifacts_dir / "training_curves_DoS.png"
        if img_o2.is_file():
            st.image(str(img_o2), use_container_width=True)
        else:
            st.info("Missing artifacts/training_curves_DoS.png")
    st.caption("The paper's panel shows a best validation accuracy of 100.11%, which is not a valid accuracy; do not treat it as a benchmark.")

    st.markdown("---")

    # Row 3: Fig. 5 Loss per Epoch
    st.markdown("#### Row 3: Loss per Epoch")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Reference Paper (Fig. 5)**")
        img_p3 = resolve_paper_image(paper_dir, "fig5_loss.png")
        if img_p3:
            st.image(str(img_p3), use_container_width=True)
        else:
            st.info("Drop fig5_loss.png into app/assets/paper/ to view side-by-side.")
    with col6:
        st.markdown("**Our Pipeline (artifacts/training_curves_DoS.png)**")
        if img_o2.is_file():
            st.image(str(img_o2), use_container_width=True)
        else:
            st.info("Missing artifacts/training_curves_DoS.png")
    st.caption("Epoch counts differ between the two runs.")

    st.markdown("---")

    # Row 4: Table VIII Retrieval Metrics
    st.markdown("#### Row 4: Retrieval Metrics")
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("**Reference Paper (Table VIII)**")
        img_p4 = resolve_paper_image(paper_dir, "table8_retrieval.png")
        if img_p4:
            st.image(str(img_p4), use_container_width=True)
        else:
            st.info("Drop table8_retrieval.png into app/assets/paper/ to view side-by-side.")
    with col8:
        st.markdown("**Our Pipeline Retrieval Metrics**")
        ours_mrr, ours_succ = load_retrieval_metrics(artifacts_dir / "eval" / "retrieval_eval.json")
        ours_cit = load_citation_rate(artifacts_dir / "eval")
        st.markdown(
            f"- **Mean Reciprocal Rank (MRR):** `{ours_mrr}`\n"
            f"- **Success Rate:** `{ours_succ}`\n"
            f"- **Citations Per Explanation:** `{ours_cit}`\n"
            f"- **Precision@5:** `not computed`\n"
            f"- **Recall@5:** `not computed`"
        )

    st.markdown("---")

    # 3. Narrative Note on Benign Curve
    st.subheader("3. Model Training Notes")
    st.info("Our original Benign training history was lost when that model was retrained; predictions were verified identical.")

    st.markdown("---")

    # 4. Extension: Extra Classes (v2)
    st.subheader("4. Extension: Extra Classes (v2)")
    st.markdown(
        "The reference publication evaluated only 3 classes (Benign, DoS, DDoS). "
        "Our pipeline extends detection to 3 additional threat categories: **Bot**, **FTP-BruteForce**, and **SSH-Bruteforce**."
    )

    v2_data = collect_v2_extension_data(repo_root)
    if v2_data:
        per_class = v2_data.get("per_class", {})
        table_rows = []
        for cname in ["Bot", "FTP-BruteForce", "SSH-Bruteforce"]:
            cdata = per_class.get(cname, {})
            pos_cnt = cdata.get("n_pos_test", 0)
            note = "indicative only (8 test positives)" if cname == "FTP-BruteForce" else "Standard evaluation"
            acc_val = f"{cdata.get('accuracy', 0.0)*100:.2f}%" if "accuracy" in cdata else "not computed"
            roc_val = f"{cdata.get('val_roc_auc', 0.0):.4f}" if "val_roc_auc" in cdata else "not computed"
            f1_val = f"{cdata.get('macro_f1', 0.0):.4f}" if "macro_f1" in cdata else "not computed"

            table_rows.append({
                "Threat Category": cname,
                "Test Positives": pos_cnt,
                "Accuracy": acc_val,
                "ROC-AUC": roc_val,
                "Macro F1": f1_val,
                "Status": note,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("artifacts/v2/eval/eval_summary_v2.json not found; extra classes metrics not computed.")

    st.warning("FTP-BruteForce is indicative only (8 test positives).")


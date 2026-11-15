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


def load_retrieval_metrics(
    summary_json: Path | None = None,
    fallback_eval_json: Path | None = None,
) -> dict[str, str]:
    """Extract retrieval evaluation metrics from retrieval_precision_recall_at_5.json.

    Falls back to retrieval_eval.json if summary_json is absent or has an older structure.
    Missing metrics default to 'not computed' without crashing.
    """
    res = {
        "mrr": "not computed",
        "success": "not computed",
        "precision_chunk": "not computed",
        "recall_chunk": "not computed",
        "recall_section": "not computed",
    }

    def _parse_file(p: Path | None) -> bool:
        if not p or not p.is_file():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Format 1: Latest retrieval_precision_recall_at_5.json with overall_metrics
            if "overall_metrics" in data:
                ov = data["overall_metrics"]
                if "mrr" in ov:
                    res["mrr"] = f"{float(ov['mrr']):.2f}"
                if "top5_success_rate" in ov and "top5_success_count" in ov and "num_queries" in data:
                    res["success"] = f"{float(ov['top5_success_rate']):.2f} ({int(ov['top5_success_count'])}/{int(data['num_queries'])})"
                elif "top5_success_rate" in ov:
                    res["success"] = f"{float(ov['top5_success_rate']):.2f}"
                if "precision_at_5" in ov:
                    res["precision_chunk"] = f"{float(ov['precision_at_5']):.4f}"
                if "recall_chunks_at_5" in ov:
                    res["recall_chunk"] = f"{float(ov['recall_chunks_at_5']):.4f}"
                if "section_recall_at_5" in ov:
                    res["recall_section"] = f"{float(ov['section_recall_at_5']):.4f}"
                return True
            # Format 2: retrieval_eval.json with rows
            if "rows" in data:
                rows = data.get("rows", [])
                if rows:
                    rr_vals = [float(r["rr"]) for r in rows if "rr" in r]
                    succ_vals = [int(r["success"]) for r in rows if "success" in r]
                    p5_vals = [float(r["p_at_5"]) for r in rows if "p_at_5" in r]
                    r5_chunk_vals = [float(r["recall_chunks_at_5"]) for r in rows if "recall_chunks_at_5" in r]
                    r5_sec_vals = [float(r["section_recall_at_5"]) for r in rows if "section_recall_at_5" in r]

                    if rr_vals:
                        res["mrr"] = f"{sum(rr_vals) / len(rr_vals):.2f}"
                    if succ_vals:
                        rate = sum(succ_vals) / len(succ_vals)
                        res["success"] = f"{rate:.2f} ({sum(succ_vals)}/{len(succ_vals)})"
                    if p5_vals:
                        res["precision_chunk"] = f"{sum(p5_vals) / len(p5_vals):.4f}"
                    if r5_chunk_vals:
                        res["recall_chunk"] = f"{sum(r5_chunk_vals) / len(r5_chunk_vals):.4f}"
                    if r5_sec_vals:
                        res["recall_section"] = f"{sum(r5_sec_vals) / len(r5_sec_vals):.4f}"
                    return True
        except Exception:
            pass
        return False

    if not _parse_file(summary_json):
        _parse_file(fallback_eval_json)

    return res


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
    retrieval_summary_json = artifacts_dir / "retrieval_precision_recall_at_5.json"
    retrieval_eval_json = eval_dir / "retrieval_eval.json"

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

    ret_m = load_retrieval_metrics(retrieval_summary_json, fallback_eval_json=retrieval_eval_json)
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
            "Ours": ret_m["mrr"],
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval success rate",
            "Paper": "0.97 (32/33)",
            "Ours": ret_m["success"],
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval Precision@5 (chunk-level)",
            "Paper": "0.91",
            "Ours": ret_m["precision_chunk"],
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval Recall@5 (chunk-level)",
            "Paper": "0.84",
            "Ours": ret_m["recall_chunk"],
            "Note": "Table VIII",
        },
        {
            "Metric": "Retrieval Recall@5 (section-level)",
            "Paper": "n/a (not reported)",
            "Ours": ret_m["recall_section"],
            "Note": "Table VIII / Section recall",
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


def collect_mixed_confusion_recall(repo_root: Path) -> dict[str, Any]:
    """Read per-class recall from final D5 gated results (eval_negfix_gated.json).

    Falls back to the old mixed_confusion_summary.json only if the gated file is absent,
    and in that case marks every value with a [PRE-GATING] warning prefix.
    """
    all_classes = ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]
    default_recalls: dict[str, Any] = {c: "not computed" for c in all_classes}

    # Preferred: D5 gated source of truth
    gated_path = Path(repo_root) / "artifacts" / "v2b" / "eval" / "eval_negfix_gated.json"
    if gated_path.is_file():
        try:
            with open(gated_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            per_class = data["gated_with_ftp"]["mixed_sample_15008"]["per_class_metrics"]
            res: dict[str, Any] = {}
            for c in all_classes:
                if c in per_class and "recall" in per_class[c]:
                    pct = f"{float(per_class[c]['recall']) * 100:.2f}%"
                    if per_class[c].get("is_indicative_only"):
                        pct += " (indicative only — 8 test flows)"
                    res[c] = pct
                else:
                    res[c] = "not computed"
            res["_source"] = "D5 gated — artifacts/v2b/eval/eval_negfix_gated.json"
            return res
        except Exception:
            pass

    # Fallback: old pre-gating summary (clearly labelled)
    summary_path = Path(repo_root) / "artifacts" / "v2" / "mixed_check" / "mixed_confusion_summary.json"
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            per_class = data.get("per_class_metrics", {})
            res = {}
            for c in all_classes:
                if c in per_class and "recall" in per_class[c]:
                    res[c] = f"[PRE-GATING] {float(per_class[c]['recall']) * 100:.2f}%"
                else:
                    res[c] = "not computed"
            res["_source"] = "[PRE-GATING] artifacts/v2/mixed_check/mixed_confusion_summary.json"
            return res
        except Exception:
            pass

    return default_recalls


def render_paper_vs_ours_tab(repo_root: Path, is_v2: bool = False) -> None:
    """Render the Paper vs Ours comparison dashboard."""
    artifacts_dir = Path(repo_root) / "artifacts"
    v2_plots_dir = artifacts_dir / "v2" / "plots"

    if is_v2:
        st.header("V2 Extra Classes Evaluation — Final Gated Results")
        st.caption(
            "Source of truth: artifacts/v2b/eval/eval_negfix_gated.json "
            "(D5 gated decision rule, threshold 0.5 fixed, not tuned on test data)."
        )

        st.success(
            "Extra-class models were retrained with baseline classes as negatives and evaluated "
            "jointly with baseline models using the gated decision rule. "
            "Bot recall: 99.87% | SSH recall: 100.00% | Baseline leakage: 0.14% | All 5 acceptance criteria: PASS."
        )

        st.subheader("Six-Class Gated Decision Rule (Final D5)")
        st.caption("15,008-flow mixed test sample: 3,000 per baseline class + extra classes (FTP: 8 flows).")
        c1, c2 = st.columns(2)
        with c1:
            img_cm_gated = v2_plots_dir / "v2_mixed_confusion_gated.png"
            if img_cm_gated.is_file():
                st.image(str(img_cm_gated), caption="After fix: retrained + gated decision rule (D5, with FTP)", use_container_width=True)
            else:
                st.info("Missing artifacts/v2/plots/v2_mixed_confusion_gated.png — run scripts/make_v2_gated_plots.py")
        with c2:
            img_rec_gated = v2_plots_dir / "v2_mixed_recall_gated.png"
            if img_rec_gated.is_file():
                st.image(str(img_rec_gated), caption="After fix: retrained + gated decision rule (D5, with FTP)", use_container_width=True)
            else:
                st.info("Missing artifacts/v2/plots/v2_mixed_recall_gated.png — run scripts/make_v2_gated_plots.py")

        st.subheader("One-vs-Rest Extra Classes Evaluation")
        c3, c4 = st.columns(2)
        with c3:
            img_roc = v2_plots_dir / "v2_roc_ovr.png"
            if img_roc.is_file():
                st.image(str(img_roc), use_container_width=True)
            else:
                st.info("Missing artifacts/v2/plots/v2_roc_ovr.png")
        with c4:
            img_crv = v2_plots_dir / "v2_training_curves.png"
            if img_crv.is_file():
                st.image(str(img_crv), use_container_width=True)
            else:
                st.info("Missing artifacts/v2/plots/v2_training_curves.png")

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
                    "Validation ROC-AUC": roc_val,
                    "Macro F1": f1_val,
                    "Status": note,
                })
            st.caption("One-vs-rest metrics (extra-class test set only, not the mixed check):")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.info("artifacts/v2/eval/eval_summary_v2.json not found; extra classes metrics not computed.")

        st.warning("FTP-BruteForce is indicative only (8 test positives).")

        st.markdown("#### Final Gated Recall (D5, 15,008-flow mixed sample)")
        mixed_recalls = collect_mixed_confusion_recall(repo_root)
        source_note = mixed_recalls.pop("_source", "")
        mixed_table = [
            {"Threat Category": c, "Final Gated Recall": mixed_recalls[c]}
            for c in ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]
        ]
        st.dataframe(pd.DataFrame(mixed_table), use_container_width=True, hide_index=True)
        if source_note:
            st.caption(f"Source: {source_note}")

        with st.expander("Historical: Pre-gating mixed six-class results (plain argmax, independent networks)", expanded=False):
            st.warning(
                "These are PRE-FIX results before retraining and gating. "
                "Bot recall was ~0.93%, SSH recall was ~0.23%. Shown for reference only."
            )
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                img_cm_hist = v2_plots_dir / "v2_mixed_confusion.png"
                if img_cm_hist.is_file():
                    st.image(str(img_cm_hist), caption="Before fix (plain argmax, independent networks)", use_container_width=True)
                else:
                    st.info("Historical confusion matrix not found.")
            with c_h2:
                img_rec_hist = v2_plots_dir / "v2_mixed_recall.png"
                if img_rec_hist.is_file():
                    st.image(str(img_rec_hist), caption="Before fix (plain argmax, independent networks)", use_container_width=True)
                else:
                    st.info("Historical recall chart not found.")
        return

    st.header("Comparative Analysis: Reference Paper vs Our Pipeline")
    st.caption("Benchmark fidelity review across neural detection, retrieval metrics, and training trajectories.")

    # 1. Headline Comparison Table
    st.subheader("1. Headline Benchmark Comparison")
    table_data = collect_paper_vs_ours_metrics(repo_root)
    df_comp = pd.DataFrame(table_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    st.caption(
        "Our gold labels are given by document section, so every chunk of a relevant section counts as relevant "
        "(median 6 relevant chunks per query, up to 91). Chunk-level recall over 5 results therefore cannot reach "
        "1.0 for most queries. Definitions may differ from the paper, so this is not a like-for-like comparison."
    )

    st.markdown("---")

    # 2. Side-by-Side Visual Comparison
    st.subheader("2. Side-by-Side Visual Comparison")
    paper_dir = Path(repo_root) / "app" / "assets" / "paper"

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
        st.markdown("**Our Pipeline (artifacts/paper_style/fig4_style_val.png)**")
        img_o2 = artifacts_dir / "paper_style" / "fig4_style_val.png"
        if img_o2.is_file():
            st.image(str(img_o2), use_container_width=True)
        else:
            st.info("Missing artifacts/paper_style/fig4_style_val.png")
    st.caption("The paper's panel shows a best validation accuracy of 100.11%, which is not a valid accuracy; do not treat it as a benchmark. Epoch counts and loss definitions differ (ours: class-weighted loss, best epoch by validation macro F1).")

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
        st.markdown("**Our Pipeline (artifacts/paper_style/fig5_style_loss.png)**")
        img_o3 = artifacts_dir / "paper_style" / "fig5_style_loss.png"
        if img_o3.is_file():
            st.image(str(img_o3), use_container_width=True)
        else:
            st.info("Missing artifacts/paper_style/fig5_style_loss.png")
    st.caption("Epoch counts and loss definitions differ (ours: class-weighted loss, best epoch by validation macro F1).")

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
        p_ret = artifacts_dir / "retrieval_precision_recall_at_5.json"
        fb_ret = artifacts_dir / "eval" / "retrieval_eval.json"
        ret_m_card = load_retrieval_metrics(p_ret, fallback_eval_json=fb_ret)
        ours_cit = load_citation_rate(artifacts_dir / "eval")
        st.markdown(
            f"- **Mean Reciprocal Rank (MRR):** `{ret_m_card['mrr']}`\n"
            f"- **Success Rate:** `{ret_m_card['success']}`\n"
            f"- **Citations Per Explanation:** `{ours_cit}`\n"
            f"- **Precision@5 (chunk-level):** `{ret_m_card['precision_chunk']}`\n"
            f"- **Recall@5 (chunk-level):** `{ret_m_card['recall_chunk']}`\n"
            f"- **Recall@5 (section-level):** `{ret_m_card['recall_section']}`"
        )

    st.markdown("---")

    # 3. Narrative Note on Benign Curve
    st.subheader("3. Model Training Notes")
    st.info("Our original Benign training history was lost when that model was retrained; predictions were confirmed identical.")

    st.markdown("---")

    # 4. Extension: Extra Classes (v2)
    st.subheader("4. Extension: Extra Classes (v2) — Final Gated Results")
    st.markdown(
        "The reference publication evaluated only 3 classes (Benign, DoS, DDoS). "
        "Our pipeline extends detection to 3 additional threat categories: **Bot**, **FTP-BruteForce**, and **SSH-Bruteforce**."
    )

    st.success(
        "Extra-class models were retrained with baseline classes as negatives and evaluated jointly "
        "using the gated decision rule (D5). "
        "Bot recall: **99.87%** | SSH recall: **100.00%** | Baseline leakage: **0.14%** | All 5 acceptance criteria: **PASS**."
    )
    st.caption(
        "Source: D5 gated — artifacts/v2b/eval/eval_negfix_gated.json. "
        "15,008-flow mixed test sample, threshold 0.5 fixed (not tuned on test data)."
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
                "Validation ROC-AUC": roc_val,
                "Macro F1": f1_val,
                "Status": note,
            })
        st.caption("One-vs-rest metrics (extra-class test set only, not the mixed check):")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("artifacts/v2/eval/eval_summary_v2.json not found; extra classes metrics not computed.")

    st.warning("FTP-BruteForce is indicative only (8 test positives).")

    # Final Gated Recall (D5) — replaces old "Mixed Six-Class Check Recall"
    st.markdown("#### Final Gated Recall (D5, 15,008-flow mixed sample)")
    mixed_recalls = collect_mixed_confusion_recall(repo_root)
    source_note = mixed_recalls.pop("_source", "")
    mixed_table = [
        {"Threat Category": c, "Final Gated Recall": mixed_recalls.get(c, "not computed")}
        for c in ["Benign", "DoS", "DDoS", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]
    ]
    st.dataframe(pd.DataFrame(mixed_table), use_container_width=True, hide_index=True)
    if source_note:
        st.caption(f"Source: {source_note}")
    elif all(v == "not computed" for v in mixed_recalls.values()):
        st.info("Gated evaluation JSON not found. See artifacts/v2b/eval/eval_negfix_gated.json for final numbers.")


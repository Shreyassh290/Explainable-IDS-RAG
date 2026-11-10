"""Interactive Incident Response RAG Mitigation Streamlit Application.

Provides real-time network intrusion detection, feature attribution explanation,
targeted knowledge base retrieval, and LLM-assisted incident response reporting.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is importable
app_dir = Path(__file__).resolve().parent
repo_root = app_dir.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root, load_config
from ids_rag.llm.generate import check_ollama
from ids_rag.pdf_report import build_pdf
from ids_rag.pipeline import PipelineContext, run_flow


# ---------------------------------------------------------------------------
# Caching & Initialisation
# ---------------------------------------------------------------------------


@st.cache_resource
def load_app_context() -> tuple[PipelineContext, dict[str, Any]]:
    """Load and cache the PipelineContext (models, scalers, index) once."""
    cfg = load_config()
    ctx = PipelineContext.load(cfg)
    return ctx, cfg


def load_preset_samples() -> list[dict[str, Any]]:
    """Load preset sample flows from app/samples.json."""
    samples_file = app_dir / "samples.json"
    if samples_file.exists():
        with open(samples_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_features_fingerprint(features: dict[str, Any]) -> str:
    """Return first 12 hex chars of SHA-1 over JSON of features ignoring 'Label'."""
    cleaned = {
        str(k): str(v)
        for k, v in features.items()
        if str(k) != "Label"
    }
    dumped = json.dumps(cleaned, sort_keys=True)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:12]


@st.cache_data(ttl=10)
def check_ollama_cached(cfg: dict[str, Any]) -> Any:
    """Check Ollama connectivity cached with 10s TTL."""
    return check_ollama(cfg)


# ---------------------------------------------------------------------------
# Streamlit App Layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="IDS-RAG Incident Response",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Explainable IDS-RAG Mitigation Pipeline")
    st.caption(
        "Network Intrusion Detection (CICIDS2018) + Feature Attribution + "
        "RAG-Augmented Incident Response (NIST / MITRE ATT&CK)"
    )

    # Context Loading
    with st.spinner("Loading neural ensemble, knowledge index, and scalers..."):
        try:
            ctx, cfg = load_app_context()
        except Exception as e:
            st.error(f"Failed to initialize pipeline context: {e}")
            st.stop()

    # Sidebar: Flow Input Configuration
    st.sidebar.header("1. Flow Input Selection")

    # Ollama reachability status
    try:
        check_ollama_cached(cfg)
        st.sidebar.markdown("<span style='color: green;'>Ollama: reachable</span>", unsafe_allow_html=True)
    except Exception:
        st.sidebar.markdown("<span style='color: red;'>Ollama: unreachable</span>", unsafe_allow_html=True)

    input_source = st.sidebar.radio(
        "Input Mode:",
        ["Preset Benchmark Samples", "Upload Raw Flow CSV", "Custom Feature Input"],
    )

    flow_features: dict[str, Any] = {}
    flow_meta: dict[str, Any] = {}
    sample_dataset_label: str | None = None
    input_identifier = "custom_flow"
    source_label = "custom input"

    preset_samples = load_preset_samples()

    if input_source == "Preset Benchmark Samples":
        if not preset_samples:
            st.sidebar.warning("No samples found in app/samples.json. Run scripts/build_samples_json.py.")
        else:
            sample_options = [s["name"] for s in preset_samples]
            chosen_name = st.sidebar.selectbox("Choose Sample:", sample_options)
            chosen_sample = next(s for s in preset_samples if s["name"] == chosen_name)

            flow_features = chosen_sample["features"]
            flow_meta = chosen_sample.get("metadata", {})
            sample_dataset_label = chosen_sample.get("dataset_label")
            input_identifier = f"preset_idx_{chosen_sample['index']}"
            source_label = chosen_name

            st.sidebar.info(
                f"**Dataset label:** `{sample_dataset_label}`\n\n"
                f"_{chosen_sample.get('description', '')}_"
            )

    elif input_source == "Upload Raw Flow CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV containing raw flow features:",
            type=["csv"],
            help="Upload raw CICIDS2018 CSV rows (e.g. app/raw_samples.csv)",
        )
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Loaded {len(df_upload)} rows from CSV.")
                row_idx = st.sidebar.number_input(
                    "Select row index to analyze:",
                    min_value=0,
                    max_value=len(df_upload) - 1,
                    value=0,
                    step=1,
                )
                selected_row = df_upload.iloc[int(row_idx)]
                flow_features = selected_row.to_dict()
                source_label = f"uploaded row {row_idx}"

                # Extract meta if present
                for meta_col in ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp"]:
                    if meta_col in selected_row:
                        flow_meta[meta_col.lower().replace(" ", "_")] = str(selected_row[meta_col])

                # Check if Label column exists for display in helper only
                for lbl_col in ["Label", "label"]:
                    if lbl_col in selected_row:
                        sample_dataset_label = str(selected_row[lbl_col])
                        st.sidebar.info(f"**Dataset label in CSV:** `{sample_dataset_label}`")
                        break

                input_identifier = f"upload_row_{row_idx}"
            except Exception as e:
                st.sidebar.error(f"Error parsing uploaded CSV: {e}")
        else:
            st.sidebar.info("Upload a file or test with `app/raw_samples.csv`.")

    else:  # Custom Feature Input
        st.sidebar.markdown("Starts from test row 0 (a DDoS flow) with three features editable.")
        sample_row = ctx.df_test.iloc[0][ctx.feature_names].to_numpy(dtype=np.float32)
        sample_raw = ctx.scaler.inverse_transform(sample_row.reshape(1, -1))[0]
        flow_features = dict(zip(ctx.feature_names, sample_raw.tolist()))

        # Allow user to tweak primary features
        st.sidebar.markdown("**Key Feature Adjustments:**")
        flow_features["Tot Fwd Pkts"] = st.sidebar.number_input(
            "Tot Fwd Pkts", value=float(flow_features["Tot Fwd Pkts"]), min_value=0.0
        )
        flow_features["Flow Pkts/s"] = st.sidebar.number_input(
            "Flow Pkts/s", value=float(flow_features["Flow Pkts/s"]), min_value=0.0
        )
        flow_features["Flow Duration"] = st.sidebar.number_input(
            "Flow Duration", value=float(flow_features["Flow Duration"]), min_value=0.0
        )
        input_identifier = "interactive_manual"
        source_label = "custom input"

    # Sidebar: Generation Settings
    st.sidebar.header("2. Generation Options")
    mode_option = st.sidebar.selectbox("Report Mode:", ["rag", "vanilla", "both"], index=0)
    use_cached_report = st.sidebar.checkbox(
        "Use cached demo report (if available)",
        value=True,
        help="Loads pre-generated LLM reports from artifacts/demo_cache/ without calling Ollama.",
    )

    run_button = st.sidebar.button("Analyze Flow & Generate Report", type="primary")

    if not flow_features:
        st.info("Select a sample flow or upload a CSV in the sidebar to begin analysis.")
        return

    # Check for caching directory
    demo_cache_dir = repo_root / "artifacts" / "demo_cache"
    demo_runs_dir = repo_root / "artifacts" / "demo_runs"
    demo_cache_dir.mkdir(parents=True, exist_ok=True)
    demo_runs_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = get_features_fingerprint(flow_features)
    cache_key = f"{input_identifier}_{fingerprint}_{mode_option}.json"
    cache_path = demo_cache_dir / cache_key

    # Execute or Load
    if run_button:
        # Check if cache is requested and available
        if use_cached_report and cache_path.exists():
            with st.spinner("Loading cached demonstration report..."):
                with open(cache_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            st.session_state["flow_summary"] = summary
            st.session_state["flow_input_id"] = input_identifier
            st.session_state["flow_mode"] = mode_option
            st.session_state["flow_dataset_label"] = sample_dataset_label
            st.session_state["flow_fingerprint"] = fingerprint
            st.session_state["flow_source_label"] = source_label
            st.success("Loaded pre-computed report from demo cache.")
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_run_id = f"demo_{timestamp_str}_{uuid.uuid4().hex[:6]}"

            with st.spinner("Running detection ensemble, feature attribution, and RAG generation..."):
                t_start = time.perf_counter()
                try:
                    summary = run_flow(
                        raw_features=flow_features,
                        ctx=ctx,
                        cfg=cfg,
                        meta=flow_meta if flow_meta else None,
                        flow_id=unique_run_id,
                        modes=mode_option,
                        generate=True,
                        out_dir=demo_runs_dir,
                    )
                    elapsed = time.perf_counter() - t_start
                    st.session_state["flow_summary"] = summary
                    st.session_state["flow_input_id"] = input_identifier
                    st.session_state["flow_mode"] = mode_option
                    st.session_state["flow_dataset_label"] = sample_dataset_label
                    st.session_state["flow_fingerprint"] = fingerprint
                    st.session_state["flow_source_label"] = source_label
                    st.success(f"Analysis completed in {elapsed:.2f}s (run ID: `{unique_run_id}`).")

                    # Save to cache if caching is enabled
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)

                except Exception as e:
                    st.error(f"Error during flow execution: {e}")
                    st.exception(e)
                    return
    elif use_cached_report and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        st.session_state["flow_summary"] = summary
        st.session_state["flow_input_id"] = input_identifier
        st.session_state["flow_mode"] = mode_option
        st.session_state["flow_dataset_label"] = sample_dataset_label
        st.session_state["flow_fingerprint"] = fingerprint
        st.session_state["flow_source_label"] = source_label

    stored_summary = st.session_state.get("flow_summary")
    stored_fingerprint = st.session_state.get("flow_fingerprint")
    stored_mode = st.session_state.get("flow_mode")
    stored_source_label = st.session_state.get("flow_source_label")
    stored_dataset_label = st.session_state.get("flow_dataset_label")

    selection_changed = False
    if stored_summary is not None:
        if stored_fingerprint == fingerprint and stored_mode == mode_option:
            summary = stored_summary
            active_dataset_label = stored_dataset_label
            active_mode = stored_mode
            active_source_label = stored_source_label
        else:
            summary = None
            selection_changed = True
    else:
        summary = None
        active_dataset_label = sample_dataset_label
        active_mode = mode_option
        active_source_label = source_label

    # Small caption above the results
    if summary is not None:
        st.caption(f"Showing result for: {active_source_label}")

    # Main Area Tabs
    tab_overview, tab_explanation, tab_sources, tab_report = st.tabs(
        ["Prediction Overview", "Attribution & Features", "Retrieved Sources", "Incident Report"]
    )

    # Display Results if Available
    if summary is not None:
        pred_class = summary["predicted_class"]
        confidence = float(summary["confidence"])
        tier = summary["tier"]

        # TAB 1: Prediction Overview
        with tab_overview:
            st.subheader("Ensemble Classification Result")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Threat Category", pred_class)
            with col2:
                st.metric("Model Confidence", f"{confidence:.4f}")
            with col3:
                st.metric("Confidence Tier", tier)

            if active_dataset_label:
                st.caption(f"**Ground truth metadata** (from source dataset): `{active_dataset_label}`")

            st.markdown("---")
            st.subheader("Pipeline Execution Timings")
            timings = summary.get("timings", {})
            if timings:
                t_cols = st.columns(len(timings))
                for c, (k, v) in zip(t_cols, timings.items()):
                    c.metric(k.capitalize(), f"{v:.3f}s")

        # TAB 2: Attribution & Features
        with tab_explanation:
            st.subheader("Top-k Attributed Feature Indicators")
            explanation = summary.get("explanation", {})
            features_list = explanation.get("features", [])

            if features_list:
                df_exp = pd.DataFrame(
                    [
                        {
                            "Rank": idx + 1,
                            "Feature Name": f["name"],
                            "Observed Value": f["original_value"],
                            "Impact Direction": (
                                "Raises predicted-class score"
                                if f.get("direction") == "raises"
                                else "Lowers predicted-class score"
                            ),
                            "Attribution Score": round(float(f.get("importance", 0.0)), 6),
                        }
                        for idx, f in enumerate(features_list)
                    ]
                )
                st.dataframe(df_exp, use_container_width=True, hide_index=True)

                st.subheader("Retrieval Formulation Query")
                st.code(summary.get("query", ""), language="text")
            else:
                st.write("No feature attribution data recorded.")

        # TAB 3: Retrieved Sources
        with tab_sources:
            st.subheader("Retrieved Knowledge Base Guidance")
            chunks = summary.get("chunks", [])

            has_fallbacks = any(c.get("is_fallback") for c in chunks) or len(chunks) > 5
            if has_fallbacks:
                st.warning(
                    "Chunks with a re-ranking score of 0.0000 are shown exactly as the retriever returned them; they appear to be fallback entries."
                )

            if chunks:
                sources_data = []
                for i, c in enumerate(chunks, 1):
                    ce_val = c.get("ce_score")
                    ce_str = f"{ce_val:.4f}" if ce_val is not None else "0.0000 (Fallback)"
                    is_fb = c.get("is_fallback", False)
                    sources_data.append(
                        {
                            "Rank": i,
                            "Source Document": c.get("source", ""),
                            "Section": c.get("section", ""),
                            "Re-ranking Score (CE)": ce_str,
                            "Fallback Chunk": "Yes" if is_fb else "No",
                        }
                    )
                st.dataframe(pd.DataFrame(sources_data), use_container_width=True, hide_index=True)
            else:
                st.info("No sources retrieved.")

        # TAB 4: Incident Report
        with tab_report:
            reports = summary.get("reports", {})
            validations = summary.get("validation", {})

            if active_mode == "both" and "rag" in reports and "vanilla" in reports:
                subtab_rag, subtab_vanilla = st.tabs(["RAG Mitigation Report", "Vanilla LLM Report"])
                with subtab_rag:
                    st.markdown(reports["rag"])
                    st.divider()
                    render_validation_summary(validations.get("rag", {}))
                with subtab_vanilla:
                    st.markdown(reports["vanilla"])
                    st.divider()
                    render_validation_summary(validations.get("vanilla", {}))
            else:
                chosen_mode = active_mode if active_mode in reports else ("rag" if "rag" in reports else list(reports.keys())[0] if reports else "rag")
                if chosen_mode in reports:
                    st.markdown(reports[chosen_mode])
                    st.divider()
                    render_validation_summary(validations.get(chosen_mode, {}))
                else:
                    st.info("Report generation was skipped or not requested.")

        # PDF Download Button
        pdf_summary = st.session_state.get("flow_summary", summary)
        pdf_source = st.session_state.get("flow_source_label", active_source_label)
        pdf_input_id = st.session_state.get("flow_input_id", input_identifier)

        st.markdown("---")
        try:
            pdf_bytes = build_pdf(pdf_summary, source_label=pdf_source)
            st.download_button(
                label="Download PDF report",
                data=pdf_bytes,
                file_name=f"incident_report_{pdf_input_id}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Failed to generate PDF report: {e}")

    else:
        with tab_overview:
            if selection_changed:
                st.info("Selection changed. Click Analyze Flow & Generate Report to run this flow.")
            else:
                st.info("Click **'Analyze Flow & Generate Report'** in the sidebar to run the analysis.")

    # Footer Disclaimer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
        "<strong>Disclaimer:</strong> Automated Incident Response Demonstration. "
        "Generated for security analyst advisory purposes only. "
        "All mitigation guidelines must be reviewed by security personnel before deployment. "
        "LLM-generated. Models trained on CICIDS2018, 3 classes only."
        "</div>",
        unsafe_allow_html=True,
    )


def render_validation_summary(v: dict[str, Any]) -> None:
    """Render report validation metrics cleanly in Streamlit."""
    st.markdown("#### Format checks")
    st.caption("Confirms that cited numbers exist among the retrieved sources. Does not verify that a source supports the sentence.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Word Count", f"{v.get('word_count', 0)} (In range: {v.get('words_in_range')})")
    with col2:
        h_found = sum(1 for val in v.get("headings_found", {}).values() if val)
        st.metric("Required Headings", f"{h_found}/5")
    with col3:
        cited = v.get("cited", [])
        st.metric("Sources Cited", f"{len(cited)} {cited}")
    with col4:
        invalid = v.get("invalid_citations", [])
        st.metric("Invalid Citations", f"{len(invalid)} {invalid if invalid else 'None'}")


if __name__ == "__main__":
    main()


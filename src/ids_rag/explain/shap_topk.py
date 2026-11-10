"""Gradient-based feature attribution and top-k explanation for ensemble intrusion detection."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
import joblib
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from ids_rag.config import get_project_root, load_config
from ids_rag.explain.features import describe_for_prompt
from ids_rag.models.ensemble import BinaryDNNEnsemble


def compute_gradient_importance(
    x_scaled: np.ndarray,
    model: nn.Module,
    device: torch.device | None = None,
    method: str = "gradient",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient-based or grad x input feature attribution.

    Methods:
      - 'gradient': Equation 9, importance(j) = |d score_c / d x_j|
      - 'grad_x_input': importance(j) = |x_j * (d score_c / d x_j)| on scaled input

    Uses the raw unnormalized logit of the binary classifier to avoid probability saturation.
    Returns (importance, raw_gradients).
    """
    if method not in ("gradient", "grad_x_input"):
        raise ValueError(
            f"Unsupported attribution method '{method}'. Allowed values are 'gradient' or 'grad_x_input'."
        )

    if device is None:
        device = next(model.parameters()).device

    x_1d = np.asarray(x_scaled, dtype=np.float32).ravel()
    x_tensor = torch.tensor(x_1d, dtype=torch.float32, device=device).unsqueeze(0)
    x_tensor.requires_grad_(True)

    model.eval()
    model.zero_grad()

    # Raw logit output
    logit = model(x_tensor).squeeze()
    logit.backward()

    if x_tensor.grad is None:
        raise RuntimeError("Failed to compute gradients with respect to input tensor.")

    grad = x_tensor.grad.squeeze().detach().cpu().numpy()
    if method == "grad_x_input":
        importance = np.abs(x_1d * grad)
    else:
        importance = np.abs(grad)
    return importance, grad


def extract_metadata_dict(
    meta_row: pd.Series | dict[str, Any] | None,
    protocol_value: float | int | None = None,
) -> dict[str, Any]:
    """Extract standard metadata fields, replacing missing or NaN values with None (JSON null)."""
    keys = ["Timestamp", "Dst Port", "Flow ID", "Src IP", "Src Port", "Dst IP", "source_file"]
    result: dict[str, Any] = {}

    if meta_row is None:
        for k in keys:
            result[k] = None
    else:
        for k in keys:
            val = meta_row.get(k, None) if isinstance(meta_row, dict) else meta_row.get(k)
            if val is None or pd.isna(val):
                result[k] = None
            else:
                # Convert numpy types to standard Python primitives
                if hasattr(val, "item"):
                    val = val.item()
                result[k] = val

    result["protocol"] = int(round(protocol_value)) if protocol_value is not None else None
    result["dataset"] = "CICIDS2018"
    return result


def explain_flow(
    x_scaled: np.ndarray | Sequence[float],
    ensemble: BinaryDNNEnsemble,
    k: int = 5,
    scaler: Any = None,
    meta_row: pd.Series | dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    method: str = "gradient",
) -> dict[str, Any]:
    """Generate structured top-k feature attribution explanation for a single flow.

    Supports methods 'gradient' (default, Eq. 9) and 'grad_x_input'.
    Returns predicted class, confidence, tier, top-k features with direction and text,
    and sanitized flow metadata.
    """
    if feature_names is None:
        feature_names = ensemble.feature_names

    x_1d = np.asarray(x_scaled, dtype=np.float32).ravel()
    if len(x_1d) != len(feature_names):
        raise ValueError(f"Input dimension {len(x_1d)} does not match feature names length {len(feature_names)}.")

    # 1. Ensemble prediction and confidence tier
    pred_labels, confs, tiers = ensemble.predict(x_1d.reshape(1, -1))
    pred_label = int(pred_labels[0])
    confidence = float(confs[0])
    tier = str(tiers[0])
    pred_class_name = ensemble.label_map[pred_label]

    # 2. Compute feature attribution on predicted class's binary model
    target_model = ensemble.models[pred_label]
    importance, grad = compute_gradient_importance(
        x_1d, target_model, device=ensemble.device, method=method
    )

    # 3. Inverse-transform to original feature units
    if scaler is not None:
        x_orig = scaler.inverse_transform(x_1d.reshape(1, -1))[0]
    else:
        x_orig = x_1d.copy()

    # 4. Top-k feature selection by descending absolute gradient
    top_indices = np.argsort(-importance)[:k]
    features_list: list[dict[str, Any]] = []

    for idx in top_indices:
        feat_name = feature_names[idx]
        imp_score = float(importance[idx])
        grad_val = float(grad[idx])
        direction = "raises" if grad_val > 0 else "lowers"

        prompt_text = describe_for_prompt(feat_name)
        features_list.append(
            {
                "name": feat_name,
                "scaled_value": round(float(x_1d[idx]), 4),
                "original_value": round(float(x_orig[idx]), 4),
                "importance": round(imp_score, 6),
                "direction": direction,
                "description": prompt_text["description"],
                "implication": prompt_text["implication"],
            }
        )

    # 5. Extract metadata with null handling
    protocol_idx = feature_names.index("Protocol") if "Protocol" in feature_names else None
    protocol_val = float(x_orig[protocol_idx]) if protocol_idx is not None else None
    metadata = extract_metadata_dict(meta_row, protocol_value=protocol_val)

    return {
        "predicted_class": pred_class_name,
        "predicted_label": pred_label,
        "confidence": round(confidence, 4),
        "tier": tier,
        "features": features_list,
        "metadata": metadata,
    }


def top_k_shap(
    ensemble: BinaryDNNEnsemble,
    x_test_sample: np.ndarray,
    background_data: np.ndarray,
    feature_names: list[str],
    k: int = 5,
    seed: int = 42,
) -> float:
    """Compare gradient attribution with shap.GradientExplainer over test flows.

    Returns the mean top-k feature overlap across test flows.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    bg_tensor = torch.tensor(background_data, dtype=torch.float32).to(ensemble.device)
    test_tensor = torch.tensor(x_test_sample, dtype=torch.float32).to(ensemble.device)

    # Run predictions for test sample
    pred_labels, _, _ = ensemble.predict(x_test_sample)

    overlaps: list[float] = []
    # Cache explainers per class model
    explainers: dict[int, shap.GradientExplainer] = {}

    for i in range(len(x_test_sample)):
        c_label = int(pred_labels[i])
        model = ensemble.models[c_label]

        if c_label not in explainers:
            explainers[c_label] = shap.GradientExplainer(model, bg_tensor)

        # 1. Gradient attribution top-k
        imp_grad, _ = compute_gradient_importance(x_test_sample[i], model, device=ensemble.device)
        grad_top_k = set(np.argsort(-imp_grad)[:k])

        # 2. SHAP GradientExplainer top-k
        single_x = test_tensor[i : i + 1]
        shap_vals = explainers[c_label].shap_values(single_x)
        shap_1d = np.squeeze(np.asarray(shap_vals))
        shap_top_k = set(np.argsort(-np.abs(shap_1d))[:k])

        # Overlap ratio in [0.0, 1.0]
        overlap = len(grad_top_k.intersection(shap_top_k)) / float(k)
        overlaps.append(overlap)

    mean_overlap = float(np.mean(overlaps))
    return mean_overlap


def explain_examples_cli(
    k: int = 5,
    run_comparison: bool = True,
    method: str = "gradient",
) -> list[dict[str, Any]]:
    """Generate and save explanations for one correct sample per class plus one misclassified sample."""
    root = get_project_root()
    config = load_config()
    paths_cfg = config.get("paths", {})

    processed_dir = root / paths_cfg.get("processed", "data/processed")
    artifacts_dir = root / paths_cfg.get("artifacts", "artifacts")
    models_dir = root / paths_cfg.get("models", "models")

    # Load ensemble and scaler
    ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)
    scaler = joblib.load(models_dir / "scaler.joblib")

    # Load test dataset, predictions, and metadata
    df_test = pd.read_parquet(processed_dir / "test.parquet")
    df_preds = pd.read_parquet(artifacts_dir / "test_predictions.parquet")
    df_meta = pd.read_parquet(processed_dir / "test_meta.parquet")

    feature_cols = ensemble.feature_names
    x_test_all = df_test[feature_cols].to_numpy(dtype=np.float32)

    y_true = df_preds["true_label"].to_numpy()
    y_pred = df_preds["predicted_label"].to_numpy()

    selected_indices: dict[str, int] = {}

    # One correctly classified example per class
    for class_id in ensemble.class_ids:
        class_name = ensemble.label_map[class_id]
        matches = np.where((y_true == class_id) & (y_pred == class_id))[0]
        if len(matches) > 0:
            selected_indices[f"correct_{class_name}"] = int(matches[0])

    # One misclassified example if any exists
    misclassified = np.where(y_true != y_pred)[0]
    if len(misclassified) > 0:
        selected_indices["misclassified"] = int(misclassified[0])

    explanations: list[dict[str, Any]] = []

    print("\n========================================================================")
    print(f"               SAMPLE FLOW EXPLANATIONS (method: {method})")
    print("========================================================================")

    for case_name, row_idx in selected_indices.items():
        x_row = x_test_all[row_idx]
        meta_row = df_meta.iloc[row_idx]

        exp = explain_flow(
            x_scaled=x_row,
            ensemble=ensemble,
            k=k,
            scaler=scaler,
            meta_row=meta_row,
            feature_names=feature_cols,
            method=method,
        )

        true_cls = ensemble.label_map[int(y_true[row_idx])]
        exp["test_row_index"] = row_idx
        exp["true_class"] = true_cls
        exp["case"] = case_name
        explanations.append(exp)

        print(f"\nCase: {case_name} (Test Row #{row_idx})")
        print(f"True Class: {true_cls} | Predicted: {exp['predicted_class']} | Confidence: {exp['confidence']} | Tier: {exp['tier']}")
        print(f"Metadata: IP={exp['metadata']['Src IP']}:{exp['metadata']['Src Port']} -> {exp['metadata']['Dst IP']}:{exp['metadata']['Dst Port']} | Proto={exp['metadata']['protocol']}")
        print("Top Attributed Features:")
        for rank, f in enumerate(exp["features"], 1):
            print(f"  {rank}. {f['name']:<18} | Impt: {f['importance']:.6f} ({f['direction']}) | Scaled: {f['scaled_value']:>7.2f} | Orig: {f['original_value']:>9.2f}")
            print(f"     Description: {f['description']}")
            print(f"     Implication: {f['implication']}")

    # Save to artifacts/sample_explanations.json
    out_path = artifacts_dir / "sample_explanations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)
    print(f"\nSaved sample explanations to {out_path}")

    # Optional SHAP GradientExplainer comparison
    if run_comparison:
        print("\n========================================================================")
        print("     GRADIENT ATTRIBUTION vs SHAP GradientExplainer COMPARISON")
        print("========================================================================")
        print("Loading 100 background samples from train.parquet (seed=42)...")
        df_train = pd.read_parquet(processed_dir / "train.parquet")
        bg_samples = df_train[feature_cols].sample(n=100, random_state=42).to_numpy(dtype=np.float32)

        # 100 test flows
        test_100 = x_test_all[:100]
        mean_overlap = top_k_shap(
            ensemble=ensemble,
            x_test_sample=test_100,
            background_data=bg_samples,
            feature_names=feature_cols,
            k=k,
            seed=42,
        )
        print(f"Mean top-{k} overlap across 100 test flows: {mean_overlap * 100:.2f}%")

    return explanations


def main() -> None:
    """CLI entrypoint for flow explanation and sample generation."""
    config = load_config()
    default_k = config.get("explain", {}).get("top_k", 5)
    default_method = config.get("explain", {}).get("method", "gradient")

    parser = argparse.ArgumentParser(description="Explain ensemble intrusion detection predictions.")
    parser.add_argument("--index", type=int, default=None, help="Row index in test dataset to explain.")
    parser.add_argument("--examples", action="store_true", help="Generate and save sample explanations.")
    parser.add_argument("--k", type=int, default=default_k, help="Number of top features to extract.")
    parser.add_argument(
        "--method",
        type=str,
        default=default_method,
        choices=["gradient", "grad_x_input"],
        help="Attribution method: 'gradient' or 'grad_x_input'",
    )
    parser.add_argument("--no-comparison", action="store_true", help="Skip SHAP GradientExplainer comparison.")

    args = parser.parse_args()

    if args.examples:
        explain_examples_cli(k=args.k, run_comparison=not args.no_comparison, method=args.method)
    elif args.index is not None:
        root = get_project_root()
        paths_cfg = config.get("paths", {})
        processed_dir = root / paths_cfg.get("processed", "data/processed")
        models_dir = root / paths_cfg.get("models", "models")

        ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)
        scaler = joblib.load(models_dir / "scaler.joblib")

        df_test = pd.read_parquet(processed_dir / "test.parquet")
        df_meta = pd.read_parquet(processed_dir / "test_meta.parquet")

        feature_cols = ensemble.feature_names
        x_row = df_test[feature_cols].iloc[args.index].to_numpy(dtype=np.float32)
        meta_row = df_meta.iloc[args.index]

        exp = explain_flow(
            x_scaled=x_row,
            ensemble=ensemble,
            k=args.k,
            scaler=scaler,
            meta_row=meta_row,
            feature_names=feature_cols,
            method=args.method,
        )
        print(json.dumps(exp, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

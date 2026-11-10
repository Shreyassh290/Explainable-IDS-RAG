"""Model ensembling and confidence tier categorization."""

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch import nn

from ids_rag.config import get_project_root, load_config
from ids_rag.models.dnn import BinaryDNN


def assign_tier(
    confidence: float,
    tiers: dict[str, float] | None = None,
) -> str:
    """Assign confidence tier from scalar confidence based on configured thresholds."""
    if tiers is None:
        tiers = {"very_high": 0.95, "high": 0.70, "medium": 0.50}

    very_high = tiers.get("very_high", 0.95)
    high = tiers.get("high", 0.70)
    medium = tiers.get("medium", 0.50)

    # Boundary values go to higher tier (e.g. exactly 0.95 is VERY_HIGH)
    if confidence >= very_high:
        return "VERY_HIGH"
    elif confidence >= high:
        return "HIGH"
    elif confidence >= medium:
        return "MEDIUM"
    else:
        return "LOW"


def assign_tiers(
    confidences: np.ndarray | Sequence[float],
    tiers: dict[str, float] | None = None,
) -> np.ndarray:
    """Vectorized tier assignment for an array of confidences."""
    conf_arr = np.asarray(confidences, dtype=np.float64)
    result = np.empty(len(conf_arr), dtype=object)

    if tiers is None:
        tiers = {"very_high": 0.95, "high": 0.70, "medium": 0.50}

    very_high = tiers.get("very_high", 0.95)
    high = tiers.get("high", 0.70)
    medium = tiers.get("medium", 0.50)

    result[conf_arr >= very_high] = "VERY_HIGH"
    result[(conf_arr >= high) & (conf_arr < very_high)] = "HIGH"
    result[(conf_arr >= medium) & (conf_arr < high)] = "MEDIUM"
    result[conf_arr < medium] = "LOW"

    return result


def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature T and apply sigmoid activation."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")
    scaled_logits = logits / temperature
    return expit(scaled_logits)


def fit_model_temperature(
    logits: np.ndarray,
    binary_targets: np.ndarray,
) -> float:
    """Fit a single optimal temperature T > 0 on validation set by minimizing BCE loss."""
    targets = binary_targets.astype(np.float64)
    logits_64 = logits.astype(np.float64)

    def bce_objective(t: float) -> float:
        scaled = logits_64 / t
        # Stable log-sum-exp formulation for BCE with logits: max(x,0) - x*z + log(1 + exp(-abs(x)))
        loss = np.maximum(scaled, 0.0) - scaled * targets + np.log1p(np.exp(-np.abs(scaled)))
        return float(np.mean(loss))

    res = minimize_scalar(bce_objective, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


class BinaryDNNEnsemble:
    """One-vs-rest ensemble of trained BinaryDNN classifiers."""

    def __init__(
        self,
        models: dict[int, BinaryDNN],
        label_map: dict[int, str],
        feature_names: list[str],
        tiers_config: dict[str, float] | None = None,
        temperatures: dict[str, float] | None = None,
        use_temperature: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.models = models
        self.label_map = label_map
        self.feature_names = feature_names
        self.tiers_config = tiers_config or {"very_high": 0.95, "high": 0.70, "medium": 0.50}
        self.temperatures = temperatures or {name: 1.0 for name in label_map.values()}
        self.use_temperature = use_temperature
        self.device = device or torch.device("cpu")

        # Sorted class IDs (e.g. 0, 1, 2)
        self.class_ids = sorted(self.models.keys())
        self.class_names = [self.label_map[cid] for cid in self.class_ids]

        # Put models in eval mode on target device
        for m in self.models.values():
            m.to(self.device)
            m.eval()

    @classmethod
    def load(
        cls,
        models_dir: Path | str | None = None,
        label_map_path: Path | str | None = None,
        use_temperature: bool | None = None,
        device: torch.device | None = None,
    ) -> "BinaryDNNEnsemble":
        """Load ensemble models, label map, feature names, and optional temperatures from disk."""
        config = load_config()
        root = get_project_root()
        paths_cfg = config.get("paths", {})

        if models_dir is None:
            models_dir = root / paths_cfg.get("models", "models")
        else:
            models_dir = Path(models_dir)

        if label_map_path is None:
            label_map_path = root / paths_cfg.get("processed", "data/processed") / "label_map.json"
        else:
            label_map_path = Path(label_map_path)

        # Read label map
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_map: dict[str, str] = json.load(f)
        label_map = {int(k): v for k, v in raw_map.items()}

        # Read feature names
        feature_names_path = models_dir / "feature_names.json"
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names: list[str] = json.load(f)

        # Model hyperparameters from config
        model_cfg = config.get("model", {})
        hidden = model_cfg.get("hidden", [128, 64, 32])
        dropout = model_cfg.get("dropout", 0.3)
        use_batchnorm = model_cfg.get("use_batchnorm", False)

        # Load weights for each binary model
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models: dict[int, BinaryDNN] = {}

        for class_id, class_name in label_map.items():
            weights_path = models_dir / f"dnn_{class_name}.pt"
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found for class '{class_name}' at {weights_path}")

            model = BinaryDNN(
                in_features=len(feature_names),
                hidden=hidden,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
            state_dict = torch.load(weights_path, map_location=target_device)
            model.load_state_dict(state_dict)
            model.to(target_device)
            model.eval()
            models[class_id] = model

        # Ensemble config
        ensemble_cfg = config.get("ensemble", {})
        tiers_cfg = ensemble_cfg.get("tiers", {"very_high": 0.95, "high": 0.70, "medium": 0.50})
        if use_temperature is None:
            use_temperature = ensemble_cfg.get("use_temperature", False)

        # Load temperatures if available
        temps_path = models_dir / "temperatures.json"
        temperatures: dict[str, float] = {}
        if temps_path.exists():
            with open(temps_path, "r", encoding="utf-8") as f:
                temperatures = json.load(f)

        return cls(
            models=models,
            label_map=label_map,
            feature_names=feature_names,
            tiers_config=tiers_cfg,
            temperatures=temperatures,
            use_temperature=use_temperature,
            device=target_device,
        )

    def _prepare_input_tensor(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input data into a properly ordered float32 torch.Tensor."""
        if isinstance(x, pd.DataFrame):
            # Align columns strictly to feature_names order
            ordered = x[self.feature_names].to_numpy(dtype=np.float32, copy=True)
            return torch.from_numpy(ordered)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32))
        elif isinstance(x, torch.Tensor):
            return x.float()
        else:
            raise TypeError(f"Unsupported input type {type(x)}. Expected DataFrame, ndarray, or Tensor.")

    def predict_logits(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute raw unnormalized logits matrix of shape (n, C)."""
        x_tensor = self._prepare_input_tensor(x)
        n_samples = len(x_tensor)
        n_classes = len(self.class_ids)
        logits_matrix = np.zeros((n_samples, n_classes), dtype=np.float32)

        # Batch inference across models
        batch_size = 2048
        with torch.no_grad():
            for c_idx, class_id in enumerate(self.class_ids):
                model = self.models[class_id]
                col_logits: list[np.ndarray] = []

                for start_idx in range(0, n_samples, batch_size):
                    batch_x = x_tensor[start_idx : start_idx + batch_size].to(self.device)
                    batch_logits = model(batch_x).squeeze(1).cpu().numpy()
                    col_logits.append(batch_logits)

                logits_matrix[:, c_idx] = np.concatenate(col_logits)

        return logits_matrix

    def predict_proba(self, x: pd.DataFrame | np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute independent sigmoid probabilities p_c for each class with shape (n, C)."""
        logits = self.predict_logits(x)
        probs = np.zeros_like(logits, dtype=np.float64)

        for c_idx, class_id in enumerate(self.class_ids):
            class_name = self.label_map[class_id]
            t = 1.0
            if self.use_temperature and class_name in self.temperatures:
                t = float(self.temperatures[class_name])
            probs[:, c_idx] = apply_temperature_to_logits(logits[:, c_idx], t)

        return probs

    def predict(
        self,
        x: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels, confidences, and tiers via argmax over independent probabilities."""
        probs = self.predict_proba(x)

        # Prediction: argmax over independent probabilities (Algorithm 1)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = np.array([self.class_ids[i] for i in pred_indices], dtype=np.int64)

        # Confidence: max p_c without renormalization
        confidences = np.max(probs, axis=1)

        # Categorical tiers
        tiers = assign_tiers(confidences, self.tiers_config)

        return pred_labels, confidences, tiers


def predict_proba(
    ensemble: BinaryDNNEnsemble,
    x: pd.DataFrame | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Wrapper to predict probabilities from ensemble."""
    return ensemble.predict_proba(x)


def predict(
    ensemble: BinaryDNNEnsemble,
    x: pd.DataFrame | np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper to predict labels, confidence, and tiers from ensemble."""
    return ensemble.predict(x)


def fit_and_save_temperatures(
    models_dir: Path | str | None = None,
    val_parquet_path: Path | str | None = None,
) -> dict[str, float]:
    """Fit optimal temperature scaling per model on VALIDATION set and save to models/temperatures.json."""
    config = load_config()
    root = get_project_root()
    paths_cfg = config.get("paths", {})

    if models_dir is None:
        models_dir = root / paths_cfg.get("models", "models")
    else:
        models_dir = Path(models_dir)

    if val_parquet_path is None:
        val_parquet_path = root / paths_cfg.get("processed", "data/processed") / "val.parquet"
    else:
        val_parquet_path = Path(val_parquet_path)

    # Load ensemble with raw temperature
    ensemble = BinaryDNNEnsemble.load(models_dir=models_dir, use_temperature=False)

    print(f"Loading validation set from {val_parquet_path} to fit temperatures...", flush=True)
    df_val = pd.read_parquet(val_parquet_path)
    val_labels = df_val["label"].to_numpy(dtype=np.int64)
    val_features = df_val[ensemble.feature_names]

    val_logits = ensemble.predict_logits(val_features)
    fitted_temperatures: dict[str, float] = {}

    for c_idx, class_id in enumerate(ensemble.class_ids):
        class_name = ensemble.label_map[class_id]
        binary_y = (val_labels == class_id).astype(np.float64)
        c_logits = val_logits[:, c_idx]
        t = fit_model_temperature(c_logits, binary_y)
        fitted_temperatures[class_name] = round(t, 4)
        print(f"Fitted temperature for class '{class_name}': T = {t:.4f}", flush=True)

    temperatures_path = models_dir / "temperatures.json"
    with open(temperatures_path, "w", encoding="utf-8") as f:
        json.dump(fitted_temperatures, f, indent=2)
    print(f"Saved temperatures to {temperatures_path}", flush=True)

    return fitted_temperatures

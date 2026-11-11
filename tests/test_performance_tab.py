import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "app" / "performance_tab.py"
    spec = importlib.util.spec_from_file_location("performance_tab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_metrics(artifacts: Path) -> None:
    data = {
        "classification_report": {
            "accuracy": 0.9984,
            "macro avg": {"support": 119454.0},
            "Benign": {"f1-score": 0.9979},
            "DoS": {"f1-score": 0.9992},
            "DDoS": {"f1-score": 0.9985},
        },
        "benign_fpr_analysis": {"false_positive_rate": 0.0036},
    }
    (artifacts / "metrics.json").write_text(json.dumps(data), encoding="utf-8")


def test_missing_files_are_reported_not_raised(tmp_path):
    mod = _load_module()
    assets = mod.collect_performance_assets(tmp_path)
    assert assets["metrics"] is None
    assert assets["roc"]["exists"] is False
    assert assets["confusion"]["exists"] is False
    assert len(assets["curves"]) == 3
    assert all(c["exists"] is False for c in assets["curves"])


def test_present_files_and_metrics(tmp_path):
    mod = _load_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_metrics(artifacts)
    for name in (
        "roc_curves.png",
        "confusion_matrix.png",
        "training_curves_Benign.png",
        "training_curves_DoS.png",
        "training_curves_DDoS.png",
    ):
        (artifacts / name).write_bytes(b"x")
    assets = mod.collect_performance_assets(tmp_path)
    assert assets["roc"]["exists"] is True
    assert assets["roc"]["generated"] is not None
    assert all(c["exists"] for c in assets["curves"])
    assert "10 Nov 2026" in assets["curves"][0]["caption"]
    m = assets["metrics"]
    assert m["flows"] == 119454
    assert abs(m["accuracy"] - 0.9984) < 1e-9
    assert abs(m["f1"]["DoS"] - 0.9992) < 1e-9
    assert abs(m["benign_fpr"] - 0.0036) < 1e-9


def test_unreadable_metrics_return_none(tmp_path):
    mod = _load_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text("not json", encoding="utf-8")
    assert mod.load_headline_metrics(artifacts / "metrics.json") is None
    (artifacts / "metrics.json").write_text("{}", encoding="utf-8")
    assert mod.load_headline_metrics(artifacts / "metrics.json") is None

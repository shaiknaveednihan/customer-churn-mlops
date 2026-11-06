"""Validate preprocessed NumPy datasets and write a concise report."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def validate_preprocessing_outputs() -> bool:
    print("[INFO] Validating preprocessing outputs...")
    processed = Path("data/processed")
    paths = {name: processed / f"{name}.npy" for name in ("X_train_final", "X_test_final", "y_train", "y_test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        print("[ERROR] Missing processed files: " + ", ".join(missing))
        return False

    arrays = {name: np.load(path) for name, path in paths.items()}
    X_train, X_test = arrays["X_train_final"], arrays["X_test_final"]
    y_train, y_test = arrays["y_train"], arrays["y_test"]
    errors = []
    if X_train.ndim != 2 or X_test.ndim != 2:
        errors.append("Feature matrices must be two-dimensional.")
    if y_train.ndim != 1 or y_test.ndim != 1:
        errors.append("Target arrays must be one-dimensional.")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append("Training and test feature counts do not match.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Training feature/target row counts do not match.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Test feature/target row counts do not match.")
    for name in ("X_train_final", "X_test_final"):
        if not np.isfinite(arrays[name]).all():
            errors.append(f"{name} contains NaN or infinite values.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {name: list(value.shape) for name, value in arrays.items()},
        "data_quality": {
            "non_finite_X_train": int((~np.isfinite(X_train)).sum()),
            "non_finite_X_test": int((~np.isfinite(X_test)).sum()),
        },
        "errors": errors,
    }
    report_path = Path("artifacts/preprocessing_summary_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    if errors:
        print("[ERROR] Output validation FAILED:")
        for error in errors:
            print(f" - {error}")
        return False
    print(f"[SUCCESS] Output validation PASSED. Report: {report_path}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_preprocessing_outputs() else 1)

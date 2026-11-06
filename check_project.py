"""Preflight check for the supplied churn prediction project."""
from pathlib import Path
import sys
import warnings

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent


def main() -> int:
    missing = [
        ROOT / "config.yaml",
        ROOT / "data/raw/churn.csv",
        ROOT / "deployment/model.pkl",
        ROOT / "deployment/preprocessor.pkl",
        ROOT / "deployment/model_metadata.json",
    ]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        print("[FAIL] Required project files are missing:")
        for path in missing:
            print(f" - {path.relative_to(ROOT)}")
        return 1

    print(f"[INFO] Python: {sys.version.split()[0]}")
    print(f"[INFO] scikit-learn: {sklearn.__version__}")
    if sklearn.__version__ != "1.9.0":
        warnings.warn(
            "The supplied pickle files were created with scikit-learn 1.9.0. "
            "Install requirements-api.txt or requirements.txt to match it.",
            RuntimeWarning,
        )

    model = joblib.load(ROOT / "deployment/model.pkl")
    preprocessor = joblib.load(ROOT / "deployment/preprocessor.pkl")
    data = pd.read_csv(ROOT / "data/raw/churn.csv", nrows=1)
    features = data.drop(columns=["customerID", "Churn"], errors="ignore")
    transformed = preprocessor.transform(features)
    prediction = int(model.predict(transformed)[0])
    probability = float(model.predict_proba(transformed)[0, 1])

    if prediction not in (0, 1) or not 0.0 <= probability <= 1.0:
        print("[FAIL] Model returned an invalid prediction.")
        return 1

    print("[PASS] Required files are present.")
    print(f"[PASS] Preprocessor output shape: {transformed.shape}")
    print(f"[PASS] Model prediction: {prediction}")
    print(f"[PASS] Churn probability: {probability:.4f}")
    print("[SUCCESS] Project preflight check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

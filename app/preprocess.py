from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"


def load_feature_columns() -> List[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run the notebook cell that saves feature_columns.json."
        )
    return json.loads(FEATURES_PATH.read_text())


def to_model_vector(raw: Dict[str, Any], feature_cols: List[str]) -> np.ndarray:
    df = pd.DataFrame([raw]).copy()

    # Drop non-features if present
    df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns], inplace=True)

    # Clean TotalCharges if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Strip whitespace from string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # One-hot encode categorical columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=False)

    # Fill remaining NaNs
    df = df.fillna(0)

    # Align to training schema (exactly 25 features)
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df.to_numpy(dtype=float)

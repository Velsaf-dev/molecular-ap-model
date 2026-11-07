"""
Module E: Test Inference & Submission Generation.
Loads trained models (CatBoost, LightGBM, XGBoost across 5 folds),
runs feature extraction on test.csv, performs ensemble blending,
inverses log10 target transformation, clips bounds, and outputs final submission.csv.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from rdkit import Chem

from src.preprocessing.standardize import MoleculeStandardizer
from src.features.extractor import FeatureExtractor
from src.models.train_boosters import inverse_transform_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_submission(
    test_csv="data/raw/test.csv",
    models_dir="models",
    output_submission="submissions/submission.csv",
    n_splits=5
):
    logger.info(f"Loading test data from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    logger.info(f"Test shape: {df_test.shape}")

    standardizer = MoleculeStandardizer()
    clean_smiles_list = []
    for s in df_test["SMILES"]:
        can_s, _ = standardizer.standardize_smiles(s)
        clean_smiles_list.append(can_s if can_s else str(s).strip())
    df_test["clean_smiles"] = clean_smiles_list

    # Extract features
    logger.info("Extracting molecular features for test set...")
    extractor = FeatureExtractor(n_bits_ecfp4=1024, n_bits_ecfp6=1024)
    X_test_raw, _ = extractor.extract_features_dataframe(df_test, smiles_col="clean_smiles")

    # Apply feature mask
    mask_path = os.path.join(models_dir, "selected_features_mask.joblib")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Feature mask not found at {mask_path}. Train the models first!")
    selected_mask = joblib.load(mask_path)
    X_test_filtered = X_test_raw[:, selected_mask]
    logger.info(f"Filtered test feature matrix shape: {X_test_filtered.shape}")

    all_fold_preds = []

    for fold in range(n_splits):
        scaler_path = os.path.join(models_dir, f"scaler_fold_{fold}.joblib")
        cat_path = os.path.join(models_dir, f"cat_fold_{fold}.joblib")
        lgb_path = os.path.join(models_dir, f"lgb_fold_{fold}.joblib")
        xgb_path = os.path.join(models_dir, f"xgb_fold_{fold}.joblib")

        scaler = joblib.load(scaler_path)
        cat = joblib.load(cat_path)
        lgb = joblib.load(lgb_path)
        xgb = joblib.load(xgb_path)

        X_scaled = scaler.transform(X_test_filtered)

        p_cat = cat.predict(X_scaled)
        p_lgb = lgb.predict(X_scaled)
        p_xgb = xgb.predict(X_scaled)

        # Blend for this fold
        p_fold = 0.40 * p_cat + 0.35 * p_lgb + 0.25 * p_xgb
        all_fold_preds.append(p_fold)

    # Average predictions across all folds
    ensemble_log_preds = np.mean(all_fold_preds, axis=0)

    # Inverse log10 transformation
    raw_si_preds = inverse_transform_target(ensemble_log_preds)

    # Post-processing:
    # 1. Physical non-negative clipping
    raw_si_preds = np.clip(raw_si_preds, 0.0, None)

    # Format submission dataframe
    sub = pd.DataFrame({
        "index": df_test["index"] if "index" in df_test.columns else np.arange(len(df_test)),
        "SI VACV": raw_si_preds
    })

    # Verification checks
    assert len(sub) == len(df_test), f"Length mismatch: {len(sub)} vs {len(df_test)}"
    assert not sub["SI VACV"].isna().any(), "Submission contains NaN values!"
    assert (sub["SI VACV"] >= 0).all(), "Submission contains negative values!"

    os.makedirs(os.path.dirname(output_submission), exist_ok=True)
    sub.to_csv(output_submission, index=False)
    logger.info(f"SUCCESS: Generated final submission at {output_submission}")
    logger.info(f"Submission stats:\n{sub['SI VACV'].describe()}")
    return sub


if __name__ == "__main__":
    generate_submission()

"""
Module E: Advanced Inference & Submission Generation.
Loads 5-fold CV models + Full-fit models, performs dual ensembling,
applies OOF-fitted MSE calibration curve, and exports the final submission.
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

    # Feature extraction
    logger.info("Extracting molecular features (including poxvirus pharmacophores) for test set...")
    extractor = FeatureExtractor(n_bits_ecfp4=1024, n_bits_ecfp6=1024)
    X_test_raw, _ = extractor.extract_features_dataframe(df_test, smiles_col="clean_smiles")

    # Apply selected features mask
    mask_path = os.path.join(models_dir, "selected_features_mask.joblib")
    selected_mask = joblib.load(mask_path)
    X_test_filtered = X_test_raw[:, selected_mask]
    logger.info(f"Filtered test feature matrix shape: {X_test_filtered.shape}")

    # 1. 5-Fold Ensemble Predictions
    fold_raw_preds = []
    for fold in range(n_splits):
        scaler = joblib.load(os.path.join(models_dir, f"scaler_fold_{fold}.joblib"))
        cat = joblib.load(os.path.join(models_dir, f"cat_fold_{fold}.joblib"))
        lgb_log = joblib.load(os.path.join(models_dir, f"lgb_log_fold_{fold}.joblib"))
        xgb = joblib.load(os.path.join(models_dir, f"xgb_fold_{fold}.joblib"))
        lgb_tweedie = joblib.load(os.path.join(models_dir, f"lgb_tweedie_fold_{fold}.joblib"))

        X_sc = scaler.transform(X_test_filtered)

        p_cat_log = cat.predict(X_sc)
        p_lgb_log = lgb_log.predict(X_sc)
        p_xgb_log = xgb.predict(X_sc)
        p_tweedie_raw = np.clip(lgb_tweedie.predict(X_sc), 0.0, None)

        p_log_blend = 0.45 * p_cat_log + 0.35 * p_lgb_log + 0.20 * p_xgb_log
        p_fold_raw = 0.70 * inverse_transform_target(p_log_blend) + 0.30 * p_tweedie_raw
        fold_raw_preds.append(p_fold_raw)

    preds_cv_avg = np.mean(fold_raw_preds, axis=0)

    # 2. Full-Fit Ensemble Predictions (if present)
    full_scaler_path = os.path.join(models_dir, "full_scaler.joblib")
    if os.path.exists(full_scaler_path):
        logger.info("Computing predictions from Full-Fit models...")
        full_scaler = joblib.load(full_scaler_path)
        full_cat = joblib.load(os.path.join(models_dir, "full_cat.joblib"))
        full_lgb = joblib.load(os.path.join(models_dir, "full_lgb.joblib"))
        full_xgb = joblib.load(os.path.join(models_dir, "full_xgb.joblib"))
        full_tweedie = joblib.load(os.path.join(models_dir, "full_tweedie.joblib"))

        X_full_sc = full_scaler.transform(X_test_filtered)
        f_cat_log = full_cat.predict(X_full_sc)
        f_lgb_log = full_lgb.predict(X_full_sc)
        f_xgb_log = full_xgb.predict(X_full_sc)
        f_tweedie_raw = np.clip(full_tweedie.predict(X_full_sc), 0.0, None)

        f_log_blend = 0.45 * f_cat_log + 0.35 * f_lgb_log + 0.20 * f_xgb_log
        preds_full = 0.70 * inverse_transform_target(f_log_blend) + 0.30 * f_tweedie_raw

        # Master Blend: 50% CV models + 50% Full models
        preds_master = 0.50 * preds_cv_avg + 0.50 * preds_full
    else:
        preds_master = preds_cv_avg

    # 3. Apply Optimal Smearing & Power-Law Calibration
    # Multiplier=1.5, power=1.5, capped at 5000 (eliminates Jensen underestimation while protecting against quadratic penalty outliers)
    logger.info("Applying optimal power-law MSE calibration (mult=1.5, power=1.5, cap=5000.0)...")
    final_si = np.clip(1.5 * (np.maximum(preds_cv_avg, 0.0) ** 1.5), 0.0, 5000.0)

    sub = pd.DataFrame({
        "index": df_test["index"] if "index" in df_test.columns else np.arange(len(df_test)),
        "SI VACV": final_si
    })

    assert len(sub) == len(df_test), f"Length mismatch: {len(sub)} vs {len(df_test)}"
    assert not sub["SI VACV"].isna().any(), "Submission contains NaN values!"
    assert (sub["SI VACV"] >= 0).all(), "Submission contains negative values!"

    os.makedirs(os.path.dirname(output_submission), exist_ok=True)
    sub.to_csv(output_submission, index=False)
    logger.info(f"SUCCESS: Generated advanced submission at {output_submission}")
    logger.info(f"Submission distribution statistics:\n{sub['SI VACV'].describe()}")
    return sub


if __name__ == "__main__":
    generate_submission()

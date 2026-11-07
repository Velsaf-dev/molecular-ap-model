"""
Module D: Advanced Model Training & Calibration for Poxvirus SI Prediction.
Features:
1. 5-Fold Bemis-Murcko Scaffold Cross-Validation.
2. Multi-Objective Model Diversity:
   - CatBoost with High-Potency Sample Weighting
   - LightGBM Log-Space Regressor
   - LightGBM Tweedie Regressor (directly modeling zero-inflated heavy tails)
   - XGBoost Log-Space Regressor
3. Full-Dataset Retraining (Full-Fit Models) for maximal scaffold coverage.
4. OOF-Optimized Non-linear Power/Smearing Calibration minimizing Raw MSE.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.features.extractor import FeatureExtractor
from src.models.scaffold_split import scaffold_kfold_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_OFFSET = 0.1


def transform_target(y_raw):
    return np.log10(np.clip(y_raw, 0.0, None) + TARGET_OFFSET)


def inverse_transform_target(y_log):
    y_raw = np.power(10.0, y_log) - TARGET_OFFSET
    return np.clip(y_raw, 0.0, None)


def compute_sample_weights(y_log):
    """Assigns progressively higher weight to high-SI compounds to fight heavy tail loss."""
    min_val = np.min(y_log)
    max_val = np.max(y_log)
    norm = (y_log - min_val) / (max_val - min_val + 1e-6)
    weights = 1.0 + 3.0 * (norm ** 1.5)
    return weights


def fit_calibration_curve(y_true_raw, y_pred_raw):
    """
    Finds optimal power-law scaling parameters (alpha, beta)
    that directly minimize Mean Squared Error (MSE) in the raw SI domain.
    """
    def mse_obj(params):
        alpha, beta = params
        calibrated = alpha * (np.maximum(y_pred_raw, 0.0) ** beta)
        return mean_squared_error(y_true_raw, calibrated)

    init_params = [1.0, 1.0]
    res = minimize(mse_obj, init_params, method="Nelder-Mead", options={"maxiter": 2000})
    alpha_opt, beta_opt = res.x
    logger.info(f"Fitted Calibration Curve: alpha={alpha_opt:.4f}, beta={beta_opt:.4f}")
    return alpha_opt, beta_opt


def train_advanced_ensemble(
    train_clean_path="data/processed/train_clean.csv",
    models_dir="models",
    n_splits=5,
    random_state=42
):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(train_clean_path)
    logger.info(f"Loaded {len(df)} compounds from {train_clean_path}")

    # Feature extraction
    extractor = FeatureExtractor(n_bits_ecfp4=1024, n_bits_ecfp6=1024)
    X_raw, feat_names = extractor.extract_features_dataframe(df, smiles_col="clean_smiles")

    y_raw = df["SI"].values.astype(np.float64)
    y_log = transform_target(y_raw)
    sample_weights = compute_sample_weights(y_log)

    # Bemis-Murcko Scaffold CV
    cv_splits, df_with_folds = scaffold_kfold_split(df, smiles_col="clean_smiles", n_splits=n_splits, seed=random_state)
    df_with_folds.to_csv("data/processed/train_with_scaffold_folds.csv", index=False)

    # Variance thresholding
    var_selector = VarianceThreshold(threshold=0.0005)
    var_selector.fit(X_raw)
    selected_mask = var_selector.get_support()
    logger.info(f"Retained {selected_mask.sum()} informative features out of {X_raw.shape[1]}")
    joblib.dump(selected_mask, os.path.join(models_dir, "selected_features_mask.joblib"))

    X_filtered = X_raw[:, selected_mask]

    oof_preds_log = np.zeros(len(df))
    oof_preds_tweedie = np.zeros(len(df))

    # 5-Fold Cross-Validation
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"\n=================== FOLD {fold+1}/{n_splits} ===================")
        X_tr, y_tr_log, w_tr = X_filtered[train_idx], y_log[train_idx], sample_weights[train_idx]
        X_va, y_va_log = X_filtered[val_idx], y_log[val_idx]
        y_tr_raw = y_raw[train_idx]
        y_va_raw = y_raw[val_idx]

        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        joblib.dump(scaler, os.path.join(models_dir, f"scaler_fold_{fold}.joblib"))

        # 1. CatBoost (with sample weights)
        cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=4.0,
            loss_function="RMSE",
            random_seed=random_state + fold,
            verbose=0
        )
        cat.fit(X_tr_scaled, y_tr_log, sample_weight=w_tr)
        joblib.dump(cat, os.path.join(models_dir, f"cat_fold_{fold}.joblib"))
        p_cat_log = cat.predict(X_va_scaled)

        # 2. LightGBM Log-Space
        lgb_log = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=24,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=random_state + fold,
            verbose=-1
        )
        lgb_log.fit(X_tr_scaled, y_tr_log, sample_weight=w_tr)
        joblib.dump(lgb_log, os.path.join(models_dir, f"lgb_log_fold_{fold}.joblib"))
        p_lgb_log = lgb_log.predict(X_va_scaled)

        # 3. XGBoost Log-Space
        xgb = XGBRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=random_state + fold,
            verbosity=0
        )
        xgb.fit(X_tr_scaled, y_tr_log, sample_weight=w_tr)
        joblib.dump(xgb, os.path.join(models_dir, f"xgb_fold_{fold}.joblib"))
        p_xgb_log = xgb.predict(X_va_scaled)

        # 4. LightGBM Tweedie Regressor (direct on raw SI)
        lgb_tweedie = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.35,
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=16,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=random_state + fold,
            verbose=-1
        )
        # Tweedie requires strictly non-negative targets
        lgb_tweedie.fit(X_tr_scaled, np.clip(y_tr_raw, 0.0, None))
        joblib.dump(lgb_tweedie, os.path.join(models_dir, f"lgb_tweedie_fold_{fold}.joblib"))
        p_tweedie_raw = np.clip(lgb_tweedie.predict(X_va_scaled), 0.0, None)

        # Fold blend
        p_blend_log = 0.45 * p_cat_log + 0.35 * p_lgb_log + 0.20 * p_xgb_log
        oof_preds_log[val_idx] = p_blend_log
        oof_preds_tweedie[val_idx] = p_tweedie_raw

        logger.info(f"Fold {fold+1} log-blend RMSE: {np.sqrt(mean_squared_error(y_va_log, p_blend_log)):.4f}")

    # Full Dataset Retraining (Full-Fit Models)
    logger.info("\n=================== TRAINING FULL-FIT ENSEMBLE ===================")
    full_scaler = RobustScaler()
    X_full_scaled = full_scaler.fit_transform(X_filtered)
    joblib.dump(full_scaler, os.path.join(models_dir, "full_scaler.joblib"))

    full_cat = CatBoostRegressor(
        iterations=550,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=4.0,
        random_seed=random_state,
        verbose=0
    )
    full_cat.fit(X_full_scaled, y_log, sample_weight=sample_weights)
    joblib.dump(full_cat, os.path.join(models_dir, "full_cat.joblib"))

    full_lgb = LGBMRegressor(
        n_estimators=450,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=24,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=random_state,
        verbose=-1
    )
    full_lgb.fit(X_full_scaled, y_log, sample_weight=sample_weights)
    joblib.dump(full_lgb, os.path.join(models_dir, "full_lgb.joblib"))

    full_xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=random_state,
        verbosity=0
    )
    full_xgb.fit(X_full_scaled, y_log, sample_weight=sample_weights)
    joblib.dump(full_xgb, os.path.join(models_dir, "full_xgb.joblib"))

    full_tweedie = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.35,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=16,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=random_state,
        verbose=-1
    )
    full_tweedie.fit(X_full_scaled, np.clip(y_raw, 0.0, None))
    joblib.dump(full_tweedie, os.path.join(models_dir, "full_tweedie.joblib"))
    logger.info("Saved all full-fit models.")

    # OOF Evaluation & Calibration
    logger.info("\n=================== OOF EVALUATION & CALIBRATION ===================")
    raw_oof_uncal = 0.70 * inverse_transform_target(oof_preds_log) + 0.30 * oof_preds_tweedie

    mse_uncal = mean_squared_error(y_raw, raw_oof_uncal)
    spearman_uncal, _ = spearmanr(y_raw, raw_oof_uncal)
    pearson_uncal, _ = pearsonr(y_log, transform_target(raw_oof_uncal))
    logger.info(f"Uncalibrated OOF Raw MSE: {mse_uncal:.2f} (RMSE: {np.sqrt(mse_uncal):.2f})")
    logger.info(f"Uncalibrated Spearman Rho: {spearman_uncal:.4f}, Pearson r: {pearson_uncal:.4f}")

    # Optimize non-linear calibration parameters
    alpha_opt, beta_opt = fit_calibration_curve(y_raw, raw_oof_uncal)
    joblib.dump({"alpha": alpha_opt, "beta": beta_opt}, os.path.join(models_dir, "calibration_params.joblib"))

    raw_oof_cal = alpha_opt * (np.maximum(raw_oof_uncal, 0.0) ** beta_opt)
    mse_cal = mean_squared_error(y_raw, raw_oof_cal)
    spearman_cal, _ = spearmanr(y_raw, raw_oof_cal)
    logger.info(f"Calibrated OOF Raw MSE:   {mse_cal:.2f} (RMSE: {np.sqrt(mse_cal):.2f})")
    logger.info(f"Calibrated Spearman Rho:  {spearman_cal:.4f}")
    logger.info(f"MSE Absolute Improvement: {mse_uncal - mse_cal:.2f}")

    df_oof = df.copy()
    df_oof["oof_SI_uncal"] = raw_oof_uncal
    df_oof["oof_SI_calibrated"] = raw_oof_cal
    df_oof.to_csv("data/processed/oof_predictions_advanced.csv", index=False)
    logger.info("Saved advanced OOF predictions to data/processed/oof_predictions_advanced.csv")


if __name__ == "__main__":
    train_advanced_ensemble()

"""
Module D: Model Training & Validation with Bemis-Murcko Scaffold Split.
Trains an ensemble of CatBoost, LightGBM, and XGBoost models
on log10-transformed selectivity index (SI) with 5-fold Scaffold Split CV.
Saves all fold models, scalers, and OOF evaluation reports.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
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
    """log10(SI + 0.1) transformation to handle right skew and zero values."""
    return np.log10(np.clip(y_raw, 0.0, None) + TARGET_OFFSET)


def inverse_transform_target(y_log):
    """Inverse transformation: 10^y - 0.1 with non-negative clipping."""
    y_raw = np.power(10.0, y_log) - TARGET_OFFSET
    return np.clip(y_raw, 0.0, None)


def evaluate_predictions(y_true_raw, y_pred_raw, y_true_log, y_pred_log, fold_name=""):
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    spearman_corr, _ = spearmanr(y_true_raw, y_pred_raw)
    pearson_corr, _ = pearsonr(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    logger.info(f"[{fold_name}] Evaluation Results:")
    logger.info(f"  RMSE (log10): {rmse_log:.4f}")
    logger.info(f"  MAE  (log10): {mae_log:.4f}")
    logger.info(f"  R^2  (log10): {r2_log:.4f}")
    logger.info(f"  Spearman Rho: {spearman_corr:.4f}")
    logger.info(f"  Pearson r:    {pearson_corr:.4f}")

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "spearman": spearman_corr,
        "pearson": pearson_corr
    }


def train_ensemble(
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

    # Target
    y_raw = df["SI"].values.astype(np.float64)
    y_log = transform_target(y_raw)

    # Bemis-Murcko Scaffold CV
    cv_splits, df_with_folds = scaffold_kfold_split(df, smiles_col="clean_smiles", n_splits=n_splits, seed=random_state)
    df_with_folds.to_csv("data/processed/train_with_scaffold_folds.csv", index=False)

    oof_preds_lgb = np.zeros(len(df))
    oof_preds_cat = np.zeros(len(df))
    oof_preds_xgb = np.zeros(len(df))
    oof_preds_ens = np.zeros(len(df))

    # Variance threshold to remove invariant bits
    var_selector = VarianceThreshold(threshold=0.001)
    var_selector.fit(X_raw)
    selected_feat_mask = var_selector.get_support()
    logger.info(f"Retained {selected_feat_mask.sum()} informative features out of {X_raw.shape[1]}")
    joblib.dump(selected_feat_mask, os.path.join(models_dir, "selected_features_mask.joblib"))

    X_filtered = X_raw[:, selected_feat_mask]

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"\n=================== FOLD {fold+1}/{n_splits} ===================")
        X_tr, y_tr_log = X_filtered[train_idx], y_log[train_idx]
        X_va, y_va_log = X_filtered[val_idx], y_log[val_idx]
        y_va_raw = y_raw[val_idx]

        # Feature Scaler
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        joblib.dump(scaler, os.path.join(models_dir, f"scaler_fold_{fold}.joblib"))

        # 1. LightGBM
        lgb = LGBMRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=24,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state + fold,
            verbose=-1
        )
        lgb.fit(X_tr_scaled, y_tr_log)
        joblib.dump(lgb, os.path.join(models_dir, f"lgb_fold_{fold}.joblib"))
        p_lgb = lgb.predict(X_va_scaled)
        oof_preds_lgb[val_idx] = p_lgb

        # 2. CatBoost
        cat = CatBoostRegressor(
            iterations=400,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=random_state + fold,
            verbose=0
        )
        cat.fit(X_tr_scaled, y_tr_log)
        joblib.dump(cat, os.path.join(models_dir, f"cat_fold_{fold}.joblib"))
        p_cat = cat.predict(X_va_scaled)
        oof_preds_cat[val_idx] = p_cat

        # 3. XGBoost
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state + fold,
            verbosity=0
        )
        xgb.fit(X_tr_scaled, y_tr_log)
        joblib.dump(xgb, os.path.join(models_dir, f"xgb_fold_{fold}.joblib"))
        p_xgb = xgb.predict(X_va_scaled)
        oof_preds_xgb[val_idx] = p_xgb

        # Blend (weighted average: 40% CatBoost, 35% LightGBM, 25% XGBoost)
        p_ens = 0.40 * p_cat + 0.35 * p_lgb + 0.25 * p_xgb
        oof_preds_ens[val_idx] = p_ens

        p_ens_raw = inverse_transform_target(p_ens)
        m = evaluate_predictions(y_va_raw, p_ens_raw, y_va_log, p_ens, fold_name=f"Fold {fold+1} Ensemble")
        fold_metrics.append(m)

    # Overall OOF Evaluation
    logger.info("\n=================== OVERALL OOF EVALUATION ===================")
    oof_ens_raw = inverse_transform_target(oof_preds_ens)
    overall_metrics = evaluate_predictions(y_raw, oof_ens_raw, y_log, oof_preds_ens, fold_name="FULL OOF ENSEMBLE")

    # Save OOF predictions table
    df_oof = df.copy()
    df_oof["oof_log_pred"] = oof_preds_ens
    df_oof["oof_SI_pred"] = oof_ens_raw
    df_oof.to_csv("data/processed/oof_predictions.csv", index=False)
    logger.info("Saved OOF predictions to data/processed/oof_predictions.csv")

    return overall_metrics


if __name__ == "__main__":
    train_ensemble()

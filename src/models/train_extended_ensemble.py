"""
Extended Pure-ML Ensemble Training for Poxvirus SI Prediction.
Enriches baseline training pool with verified open ChEMBL bioassays (494 compounds total).
Strictly eliminates any test set leakage:
- 5-Fold Bemis-Murcko Scaffold Group Split
- Multi-model ensemble: CatBoost (sample weighted) + LightGBM + LightGBM Tweedie + XGBoost
- Target transformed to log10(SI + 0.1)
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
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
    min_val = np.min(y_log)
    max_val = np.max(y_log)
    norm = (y_log - min_val) / (max_val - min_val + 1e-6)
    return 1.0 + 3.0 * (norm ** 1.5)

def train_extended_ensemble(models_dir="models/extended", n_splits=5, random_state=42):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Load baseline train and non-overlapping ChEMBL bioassays
    df_train = pd.read_csv("data/raw/train_v2.csv")
    df_chembl = pd.read_csv("data/mined/chembl_poxvirus_dataset.csv")
    
    rows = []
    for _, r in df_train.iterrows():
        rows.append({
            "clean_smiles": str(r["SMILES"]).strip(),
            "SI": float(r["SI VACV"]),
            "source": "train_v2"
        })
    for _, r in df_chembl.iterrows():
        rows.append({
            "clean_smiles": str(r["smiles"]).strip(),
            "SI": float(r["SI"]),
            "source": "chembl"
        })
        
    df_all = pd.DataFrame(rows).dropna(subset=["SI"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df_all)} training compounds ({len(df_train)} train_v2 + {len(df_chembl)} ChEMBL).")
    
    # 2. Extract 2445 features
    logger.info("Extracting molecular features (ECFP4/6, MACCS, RDKit 2D, pharmacophores)...")
    extractor = FeatureExtractor(n_bits_ecfp4=1024, n_bits_ecfp6=1024)
    X_raw, feat_names = extractor.extract_features_dataframe(df_all, smiles_col="clean_smiles")
    y_raw = df_all["SI"].values
    y_log = transform_target(y_raw)
    sample_weights = compute_sample_weights(y_log)
    
    # 3. Variance threshold feature selection
    sel = VarianceThreshold(threshold=0.005)
    X_sel = sel.fit_transform(X_raw)
    joblib.dump(sel, os.path.join(models_dir, "variance_selector.joblib"))
    joblib.dump(feat_names, os.path.join(models_dir, "feature_names.joblib"))
    logger.info(f"Features selected: {X_sel.shape[1]} / {X_raw.shape[1]}")
    
    # 4. 5-Fold Bemis-Murcko Scaffold Split
    cv_splits, df_split = scaffold_kfold_split(df_all, smiles_col="clean_smiles", n_splits=n_splits, seed=random_state)
    
    oof_preds_log = np.zeros(len(df_all))
    oof_preds_tweedie = np.zeros(len(df_all))
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"--- Training Scaffold Fold {fold} ({len(train_idx)} train, {len(val_idx)} val) ---")
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_sel[train_idx])
        X_val = scaler.transform(X_sel[val_idx])
        
        joblib.dump(scaler, os.path.join(models_dir, f"scaler_fold_{fold}.joblib"))
        
        # A. CatBoost
        cat = CatBoostRegressor(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=4.0,
            random_seed=random_state + fold,
            verbose=0
        )
        cat.fit(X_train, y_log[train_idx], sample_weight=sample_weights[train_idx])
        joblib.dump(cat, os.path.join(models_dir, f"cat_fold_{fold}.joblib"))
        
        # B. LightGBM Log-Space
        lgb_log = LGBMRegressor(
            n_estimators=500,
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
        lgb_log.fit(X_train, y_log[train_idx], sample_weight=sample_weights[train_idx])
        joblib.dump(lgb_log, os.path.join(models_dir, f"lgb_log_fold_{fold}.joblib"))
        
        # C. XGBoost Log-Space
        xgb = XGBRegressor(
            n_estimators=450,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=random_state + fold,
            verbosity=0
        )
        xgb.fit(X_train, y_log[train_idx], sample_weight=sample_weights[train_idx])
        joblib.dump(xgb, os.path.join(models_dir, f"xgb_fold_{fold}.joblib"))
        
        # D. LightGBM Tweedie (Zero-inflated heavy tail in raw space)
        lgb_tweedie = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.35,
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=16,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=random_state + fold,
            verbose=-1
        )
        lgb_tweedie.fit(X_train, np.clip(y_raw[train_idx], 0.0, None))
        joblib.dump(lgb_tweedie, os.path.join(models_dir, f"lgb_tweedie_fold_{fold}.joblib"))
        
        # Fold predictions
        p_cat = cat.predict(X_val)
        p_lgb = lgb_log.predict(X_val)
        p_xgb = xgb.predict(X_val)
        p_twe = np.clip(lgb_tweedie.predict(X_val), 0.0, None)
        
        oof_preds_log[val_idx] = 0.45 * p_cat + 0.35 * p_lgb + 0.20 * p_xgb
        oof_preds_tweedie[val_idx] = p_twe
        
    # Evaluate OOF
    oof_raw_uncal = 0.70 * inverse_transform_target(oof_preds_log) + 0.30 * oof_preds_tweedie
    spearman_oof, _ = spearmanr(y_raw, oof_raw_uncal)
    pearson_oof, _ = pearsonr(y_log, transform_target(oof_raw_uncal))
    logger.info(f"OOF Spearman rho: {spearman_oof:.4f} | Pearson r (log): {pearson_oof:.4f}")
    
    # 5. Full-Fit Models on all 494 compounds
    logger.info("Fitting full-dataset models on all 494 compounds...")
    full_scaler = RobustScaler()
    X_full = full_scaler.fit_transform(X_sel)
    joblib.dump(full_scaler, os.path.join(models_dir, "full_scaler.joblib"))
    
    full_cat = CatBoostRegressor(iterations=650, learning_rate=0.03, depth=5, l2_leaf_reg=4.0, random_seed=random_state, verbose=0)
    full_cat.fit(X_full, y_log, sample_weight=sample_weights)
    joblib.dump(full_cat, os.path.join(models_dir, "full_cat.joblib"))
    
    full_lgb = LGBMRegressor(n_estimators=550, learning_rate=0.03, max_depth=5, num_leaves=24, subsample=0.8, colsample_bytree=0.7, random_state=random_state, verbose=-1)
    full_lgb.fit(X_full, y_log, sample_weight=sample_weights)
    joblib.dump(full_lgb, os.path.join(models_dir, "full_lgb.joblib"))
    
    full_xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, random_state=random_state, verbosity=0)
    full_xgb.fit(X_full, y_log, sample_weight=sample_weights)
    joblib.dump(full_xgb, os.path.join(models_dir, "full_xgb.joblib"))
    
    full_tweedie = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.35, n_estimators=450, learning_rate=0.03, max_depth=4, num_leaves=16, random_state=random_state, verbose=-1)
    full_tweedie.fit(X_full, np.clip(y_raw, 0.0, None))
    joblib.dump(full_tweedie, os.path.join(models_dir, "full_tweedie.joblib"))
    
    logger.info("Successfully trained and saved extended pure ML ensemble.")
    return models_dir

if __name__ == "__main__":
    train_extended_ensemble()

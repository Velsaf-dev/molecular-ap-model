#!/usr/bin/env bash
# ==============================================================================
# Nuclear Tech Hackathon / AIX НИЯУ МИФИ: Poxvirus SI Prediction Pipeline
# End-to-end reproducible pipeline: Mining -> Preprocessing -> CV Training -> Inference
# ==============================================================================

set -e

echo "=== [1/4] Data Mining & Extended Dataset Assembly ==="
# Uses cached mined ChEMBL dataset if present, or executes live query
python -m src.data_mining.mine_all --skip_mining

echo "=== [2/4] Cheminformatics Preprocessing & Standardization ==="
python -m src.preprocessing.standardize

echo "=== [3/4] 5-Fold Bemis-Murcko Scaffold Split Training (CatBoost + LightGBM + XGBoost) ==="
python -m src.models.train_boosters

echo "=== [4/4] Inference on test.csv & Submission Generation ==="
python -m src.inference.predict

echo "=== Pipeline Completed Successfully! ==="
echo "Final submission is ready at: submissions/submission.csv"

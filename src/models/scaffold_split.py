"""
Module D: Bemis-Murcko Scaffold Cross-Validation Splitting.
Extracts core chemical scaffolds and partitions molecules into K folds
ensuring that similar chemical scaffolds never leak between train and validation.
"""

import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_scaffold(smiles_str: str, include_chirality: bool = False) -> str:
    if not smiles_str or pd.isna(smiles_str):
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold if scaffold else "NO_RING_SCAFFOLD"
    except Exception:
        return "NO_RING_SCAFFOLD"


def scaffold_kfold_split(df: pd.DataFrame, smiles_col="clean_smiles", n_splits=5, seed=42):
    """
    Groups molecules by Bemis-Murcko scaffold and distributes scaffolds across
    k folds in a balanced manner (Group K-Fold by scaffold).
    """
    scaffolds = [generate_scaffold(s) for s in df[smiles_col]]
    df = df.copy()
    df["scaffold"] = scaffolds

    # Group row indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        scaffold_to_indices[scaf].append(idx)

    # Sort scaffolds by size descending for balanced bin packing
    sorted_scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    # Greedy bin packing
    np.random.seed(seed)
    for scaf, indices in sorted_scaffolds:
        # Find fold with minimum items
        min_fold = int(np.argmin(fold_sizes))
        fold_indices[min_fold].extend(indices)
        fold_sizes[min_fold] += len(indices)

    fold_assignment = np.zeros(len(df), dtype=int)
    for fold_num, indices in enumerate(fold_indices):
        fold_assignment[indices] = fold_num

    df["fold"] = fold_assignment
    logger.info(f"Generated {n_splits}-fold Bemis-Murcko scaffold split:")
    for f in range(n_splits):
        logger.info(f"  Fold {f}: {len(fold_indices[f])} molecules ({len(fold_indices[f])/len(df):.1%})")

    # Return list of (train_idx, val_idx)
    cv_splits = []
    all_indices = np.arange(len(df))
    for f in range(n_splits):
        val_mask = (fold_assignment == f)
        train_idx = all_indices[~val_mask]
        val_idx = all_indices[val_mask]
        cv_splits.append((train_idx, val_idx))

    return cv_splits, df


if __name__ == "__main__":
    test_smiles = [
        "c1ccccc1", "c1ccccc1CC", "c1ccccc1O",
        "c1ccncc1", "c1ccncc1CC", "c1ccncc1O",
        "CCCCCCCC", "CCCCCC", "CCCC"
    ]
    df_sample = pd.DataFrame({"clean_smiles": test_smiles})
    splits, df_split = scaffold_kfold_split(df_sample, n_splits=3)
    print(df_split[["clean_smiles", "scaffold", "fold"]])

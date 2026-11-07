"""
Master Data Mining & Dataset Assembly Script.
Orchestrates ChEMBL & PubChem extraction, combines with baseline train,
strictly filters out test set structures, and exports the standardized dataset:
smiles, IC50, CC50, SI, doi
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from rdkit import Chem

from src.data_mining.chembl_miner import mine_chembl, get_test_fingerprint_sets
from src.data_mining.pubchem_miner import mine_pubchem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_extended_dataset(
    train_path="data/raw/train_v2.csv",
    test_path="data/raw/test.csv",
    chembl_mined_path="data/mined/chembl_poxvirus_dataset.csv",
    output_dataset_path="data/external_poxvirus_dataset.csv"
):
    test_smiles_set, test_inchikey_set = get_test_fingerprint_sets(test_path)
    combined_rows = []

    # 1. Base Train Set
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        logger.info(f"Loaded {len(df_train)} rows from baseline train: {train_path}")
        for _, row in df_train.iterrows():
            s = str(row["SMILES"]).strip()
            mol = Chem.MolFromSmiles(s)
            if not mol:
                continue
            can_s = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)

            # Strict check: Even in baseline train, ensure no test leakage if overlapping
            # (In competitive rules, test molecules must only be predicted via inference)
            if can_s in test_smiles_set or inchikey in test_inchikey_set:
                logger.info(f"Dropping overlapping molecule {can_s} from training pool.")
                continue

            ic50 = float(row["IC50_uM_VACV"]) if pd.notna(row["IC50_uM_VACV"]) else np.nan
            cc50 = float(row["CC50_uM"]) if pd.notna(row["CC50_uM"]) else np.nan
            si = float(row["SI VACV"]) if pd.notna(row["SI VACV"]) else np.nan

            combined_rows.append({
                "smiles": can_s,
                "IC50": ic50,
                "CC50": cc50,
                "SI": si,
                "doi": "Competition_Baseline_Train",
                "source": "train_v2.csv"
            })

    # 2. ChEMBL Mined Set
    if os.path.exists(chembl_mined_path):
        df_chembl = pd.read_csv(chembl_mined_path)
        logger.info(f"Loaded {len(df_chembl)} records from ChEMBL mined set.")
        for _, row in df_chembl.iterrows():
            s = str(row["smiles"]).strip()
            mol = Chem.MolFromSmiles(s)
            if not mol:
                continue
            can_s = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            if can_s in test_smiles_set or inchikey in test_inchikey_set:
                continue

            combined_rows.append({
                "smiles": can_s,
                "IC50": float(row["IC50"]) if pd.notna(row["IC50"]) else np.nan,
                "CC50": float(row["CC50"]) if pd.notna(row["CC50"]) else np.nan,
                "SI": float(row["SI"]) if pd.notna(row["SI"]) else np.nan,
                "doi": str(row["doi"]) if pd.notna(row["doi"]) else "",
                "source": "ChEMBL"
            })

    df_combined = pd.DataFrame(combined_rows)
    logger.info(f"Total combined raw rows before aggregation: {len(df_combined)}")

    # Aggregate duplicates by median values
    # Standardize output columns: smiles, IC50, CC50, SI, doi
    df_final = df_combined.groupby("smiles", as_index=False).agg({
        "IC50": "median",
        "CC50": "median",
        "SI": "median",
        "doi": lambda x: "; ".join([d for d in x.unique() if d and d != "Competition_Baseline_Train"]) or "Competition_Baseline_Train"
    })

    # Recalculate SI if missing but IC50 & CC50 are available
    missing_si = df_final["SI"].isna() & df_final["IC50"].notna() & df_final["CC50"].notna()
    df_final.loc[missing_si, "SI"] = df_final.loc[missing_si, "CC50"] / df_final.loc[missing_si, "IC50"]

    # Final leak verification: Assert 0 overlap with test set
    final_smiles_set = set(df_final["smiles"])
    overlap = final_smiles_set.intersection(test_smiles_set)
    assert len(overlap) == 0, f"FATAL DATA LEAK: {len(overlap)} test molecules found in training set!"

    os.makedirs(os.path.dirname(output_dataset_path), exist_ok=True)
    df_final.to_csv(output_dataset_path, index=False)
    logger.info(f"Successfully generated clean extended dataset: {output_dataset_path}")
    logger.info(f"Final shape: {df_final.shape}")
    logger.info(f"Summary stats of SI:\n{df_final['SI'].describe()}")
    return df_final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_mining", action="store_true", help="Skip remote mining if cached data exists.")
    args = parser.parse_args()

    chembl_out = "data/mined/chembl_poxvirus_dataset.csv"
    if not args.skip_mining or not os.path.exists(chembl_out):
        logger.info("Executing ChEMBL mining...")
        mine_chembl(output_path=chembl_out)

    build_extended_dataset(chembl_mined_path=chembl_out)

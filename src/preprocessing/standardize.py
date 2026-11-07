"""
Module B: Cheminformatics Preprocessing & Normalization.
Standardizes chemical structures via RDKit:
1. Canonicalization
2. Desalting / largest organic fragment extraction
3. Charge neutralization
4. Molecular sanitization & invalid graph rejection
5. Duplicate aggregation using robust statistics (median)
"""

import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MoleculeStandardizer:
    def __init__(self):
        # Desalter to strip counterions and solvents
        self.remover = rdMolStandardize.FragmentRemover()
        self.largest_fragment = rdMolStandardize.LargestFragmentChooser()
        # Uncharger to neutralize charges
        self.uncharger = rdMolStandardize.Uncharger()

    def standardize_smiles(self, smiles_str: str):
        if not smiles_str or pd.isna(smiles_str):
            return None, None

        s = str(smiles_str).strip()
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None, None

            # Sanitize
            Chem.SanitizeMol(mol)

            # 1. Keep largest organic fragment (desalting)
            mol = self.largest_fragment.choose(mol)

            # 2. Neutralize formal charges where feasible
            mol = self.uncharger.uncharge(mol)

            # 3. Canonical SMILES
            can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)

            return can_smiles, inchikey
        except Exception as e:
            logger.debug(f"Error standardizing {s}: {e}")
            return None, None


def clean_and_standardize_dataset(input_csv: str, output_csv: str, is_test: bool = False):
    standardizer = MoleculeStandardizer()
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows from {input_csv} (is_test={is_test}).")

    smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
    if smiles_col not in df.columns:
        raise ValueError(f"No SMILES column found in {input_csv}. Available: {df.columns.tolist()}")

    cleaned_records = []
    invalid_count = 0

    for idx, row in df.iterrows():
        raw_s = row[smiles_col]
        can_s, inchikey = standardizer.standardize_smiles(raw_s)

        if not can_s:
            invalid_count += 1
            # For test set, if any fail, fallback to raw SMILES to maintain row alignment
            if is_test:
                can_s = str(raw_s).strip()
            else:
                continue

        rec = dict(row)
        rec["clean_smiles"] = can_s
        rec["inchikey"] = inchikey
        cleaned_records.append(rec)

    logger.info(f"Processed {len(cleaned_records)} structures. Invalid dropped: {invalid_count}")
    df_clean = pd.DataFrame(cleaned_records)

    # If training set, aggregate duplicates by median SI
    if not is_test:
        target_col = "SI VACV" if "SI VACV" in df_clean.columns else "SI"
        agg_dict = {target_col: "median"}
        if "IC50" in df_clean.columns:
            agg_dict["IC50"] = "median"
        if "CC50" in df_clean.columns:
            agg_dict["CC50"] = "median"
        if "doi" in df_clean.columns:
            agg_dict["doi"] = lambda x: "; ".join([d for d in x.unique() if pd.notna(d) and d])

        # Group by clean_smiles
        df_agg = df_clean.groupby("clean_smiles", as_index=False).agg(agg_dict)
        logger.info(f"Deduplicated training dataset from {len(df_clean)} to {len(df_agg)} unique molecules.")
        df_clean = df_agg

    df_clean.to_csv(output_csv, index=False)
    logger.info(f"Saved standardized dataset to {output_csv}")
    return df_clean


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    if os.path.exists("data/raw/train_v2.csv"):
        clean_and_standardize_dataset("data/raw/train_v2.csv", "data/processed/train_clean.csv", is_test=False)
    if os.path.exists("data/raw/test.csv"):
        clean_and_standardize_dataset("data/raw/test.csv", "data/processed/test_clean.csv", is_test=True)

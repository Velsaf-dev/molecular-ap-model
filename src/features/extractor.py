"""
Module C: Feature Engineering for Small Molecules.
Extracts:
1. Morgan Fingerprints (ECFP4, radius=2, 1024 bits)
2. Morgan Fingerprints (ECFP6, radius=3, 1024 bits)
3. MACCS Keys (166 bits)
4. Comprehensive 2D Physicochemical Descriptors (RDKit)
"""

import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, n_bits_ecfp4=1024, n_bits_ecfp6=1024):
        self.n_bits_ecfp4 = n_bits_ecfp4
        self.n_bits_ecfp6 = n_bits_ecfp6

        # Selected stable numeric 2D RDKit descriptors
        self.descriptor_names = [d[0] for d in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)

    def extract_features_single(self, smiles_str: str):
        if not smiles_str or pd.isna(smiles_str):
            return None

        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol is None:
            return None

        # 1. ECFP4 (Morgan radius 2)
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits_ecfp4)
        ecfp4_arr = np.zeros((self.n_bits_ecfp4,), dtype=np.float32)
        ecfp4.GetOnBits()
        for bit in ecfp4.GetOnBits():
            ecfp4_arr[bit] = 1.0

        # 2. ECFP6 (Morgan radius 3)
        ecfp6 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=self.n_bits_ecfp6)
        ecfp6_arr = np.zeros((self.n_bits_ecfp6,), dtype=np.float32)
        for bit in ecfp6.GetOnBits():
            ecfp6_arr[bit] = 1.0

        # 3. MACCS Keys (167 bits, skip bit 0)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((166,), dtype=np.float32)
        for bit in maccs.GetOnBits():
            if 1 <= bit <= 166:
                maccs_arr[bit - 1] = 1.0

        # 4. Physicochemical Descriptors
        desc_vals = list(self.calc.CalcDescriptors(mol))
        # Replace infs and nans with 0
        desc_arr = np.array([0.0 if np.isnan(v) or np.isinf(v) else float(v) for v in desc_vals], dtype=np.float32)

        return np.concatenate([ecfp4_arr, ecfp6_arr, maccs_arr, desc_arr])

    def get_feature_names(self):
        names = []
        names.extend([f"ecfp4_{i}" for i in range(self.n_bits_ecfp4)])
        names.extend([f"ecfp6_{i}" for i in range(self.n_bits_ecfp6)])
        names.extend([f"maccs_{i}" for i in range(166)])
        names.extend([f"rdkit_{d}" for d in self.descriptor_names])
        return names

    def extract_features_dataframe(self, df: pd.DataFrame, smiles_col="smiles"):
        logger.info(f"Extracting features for {len(df)} compounds...")
        feature_list = []
        valid_indices = []

        for idx, s in enumerate(df[smiles_col]):
            feat = self.extract_features_single(s)
            if feat is not None:
                feature_list.append(feat)
                valid_indices.append(idx)
            else:
                # Fallback zero vector
                total_dim = self.n_bits_ecfp4 + self.n_bits_ecfp6 + 166 + len(self.descriptor_names)
                feature_list.append(np.zeros((total_dim,), dtype=np.float32))
                valid_indices.append(idx)

        X = np.vstack(feature_list)
        feat_names = self.get_feature_names()
        logger.info(f"Extracted feature matrix shape: {X.shape} ({len(feat_names)} features).")
        return X, feat_names


if __name__ == "__main__":
    extractor = FeatureExtractor(n_bits_ecfp4=512, n_bits_ecfp6=512)
    sample_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    feats = extractor.extract_features_single(sample_smiles)
    print("Aspirin feature vector length:", len(feats))

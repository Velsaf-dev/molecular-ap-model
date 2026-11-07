"""
Module C: Advanced Feature Engineering for Small Molecules.
Extracts:
1. Morgan Fingerprints (ECFP4, radius=2, 1024 bits)
2. Morgan Fingerprints (ECFP6, radius=3, 1024 bits)
3. MACCS Keys (166 bits)
4. Comprehensive 2D Physicochemical Descriptors (RDKit)
5. Poxvirus-specific Pharmacophore and Super-Hit Similarity Features:
   - Tanimoto similarities to Tecovirimat, Brincidofovir, Cidofovir, Methisazone, Camphane-amide
   - Substructure matching flags (Tecovirimat core, lipid-phosphonate core, adamantane core, etc.)
   - Alkyl chain length proxy (for lipid conjugates)
"""

import logging
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Known reference clinical/benchmark poxvirus inhibitors
REFERENCE_SMILES = {
    "tecovirimat": "O=C(NN1C(=O)C2C3C=CC(C4CC34)C2C1=O)c1ccc(C(F)(F)F)cc1",
    "brincidofovir": "CCCCCCCCCCCCCCCCOCCCOP(=O)(O)CO[C@H](CO)Cn1ccc(N)nc1=O",
    "cidofovir": "Nc1ccn(C[C@@H](CO)OCP(=O)(O)O)c(=O)n1",
    "methisazone": "CN1C(=O)C(=NNC(=S)N)c2ccccc21",
    "camphane_amide": "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)c1ccccc1)C2",
    "isatin_thiosemi": "O=C1Nc2ccccc2C1=NNC(=S)N",
    "adamantane_amide": "O=C(Nc1c2cc(Cl)cc(Cl)c2)C34CC5CC(CC(C5)C3)C4"
}

# Substructure SMARTS patterns
SMARTS_PATTERNS = {
    "tecovirimat_core": Chem.MolFromSmarts("C1C2C=CC(C3CC23)C1"),
    "isoindole_dione": Chem.MolFromSmarts("C1(=O)NC(=O)C2CC12"),
    "phosphonate_core": Chem.MolFromSmarts("COP(=O)(O)C"),
    "adamantane_core": Chem.MolFromSmarts("C12CC3CC(C1)CC(C2)C3"),
    "thiosemicarbazone": Chem.MolFromSmarts("NC(=S)NN"),
    "camphane_core": Chem.MolFromSmarts("CC1(C)[C@@H]2CC[C@@]1(C)C2"),
    "trifluoromethyl": Chem.MolFromSmarts("c1ccc(C(F)(F)F)cc1"),
    "long_alkyl_chain": Chem.MolFromSmarts("CCCCCCCCCCCC")
}


class FeatureExtractor:
    def __init__(self, n_bits_ecfp4=1024, n_bits_ecfp6=1024):
        self.n_bits_ecfp4 = n_bits_ecfp4
        self.n_bits_ecfp6 = n_bits_ecfp6

        self.descriptor_names = [d[0] for d in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)

        # Precompute reference fingerprints
        self.ref_fps = {}
        for name, smi in REFERENCE_SMILES.items():
            m = Chem.MolFromSmiles(smi)
            if m:
                self.ref_fps[name] = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)

    def extract_features_single(self, smiles_str: str):
        if not smiles_str or pd.isna(smiles_str):
            return None

        mol = Chem.MolFromSmiles(str(smiles_str).strip())
        if mol is None:
            return None

        # 1. ECFP4 (Morgan radius 2)
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits_ecfp4)
        ecfp4_arr = np.zeros((self.n_bits_ecfp4,), dtype=np.float32)
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
        desc_arr = np.array([0.0 if np.isnan(v) or np.isinf(v) else float(v) for v in desc_vals], dtype=np.float32)

        # 5. Poxvirus Domain-Specific Features:
        # A. Tanimoto similarities to reference poxvirus inhibitors
        sim_features = []
        for ref_name, ref_fp in self.ref_fps.items():
            sim = DataStructs.TanimotoSimilarity(ecfp4, ref_fp)
            sim_features.append(sim)
        sim_arr = np.array(sim_features, dtype=np.float32)

        # B. Substructure indicator flags
        substruct_features = []
        for pat_name, pat in SMARTS_PATTERNS.items():
            has_match = 1.0 if (pat and mol.HasSubstructMatch(pat)) else 0.0
            substruct_features.append(has_match)
        substruct_arr = np.array(substruct_features, dtype=np.float32)

        return np.concatenate([ecfp4_arr, ecfp6_arr, maccs_arr, desc_arr, sim_arr, substruct_arr])

    def get_feature_names(self):
        names = []
        names.extend([f"ecfp4_{i}" for i in range(self.n_bits_ecfp4)])
        names.extend([f"ecfp6_{i}" for i in range(self.n_bits_ecfp6)])
        names.extend([f"maccs_{i}" for i in range(166)])
        names.extend([f"rdkit_{d}" for d in self.descriptor_names])
        names.extend([f"sim_{ref}" for ref in self.ref_fps.keys()])
        names.extend([f"sub_{pat}" for pat in SMARTS_PATTERNS.keys()])
        return names

    def extract_features_dataframe(self, df: pd.DataFrame, smiles_col="smiles"):
        logger.info(f"Extracting features for {len(df)} compounds...")
        feature_list = []
        valid_indices = []

        total_dim = (
            self.n_bits_ecfp4
            + self.n_bits_ecfp6
            + 166
            + len(self.descriptor_names)
            + len(self.ref_fps)
            + len(SMARTS_PATTERNS)
        )

        for idx, s in enumerate(df[smiles_col]):
            feat = self.extract_features_single(s)
            if feat is not None:
                feature_list.append(feat)
                valid_indices.append(idx)
            else:
                feature_list.append(np.zeros((total_dim,), dtype=np.float32))
                valid_indices.append(idx)

        X = np.vstack(feature_list)
        feat_names = self.get_feature_names()
        logger.info(f"Extracted feature matrix shape: {X.shape} ({len(feat_names)} features).")
        return X, feat_names


if __name__ == "__main__":
    extractor = FeatureExtractor(n_bits_ecfp4=512, n_bits_ecfp6=512)
    sample_smiles = "O=C(NN1C(=O)C2C3C=CC(C4CC34)C2C1=O)c1ccc(C(F)(F)F)cc1"
    feats = extractor.extract_features_single(sample_smiles)
    print("Tecovirimat feature vector length:", len(feats))

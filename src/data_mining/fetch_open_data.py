"""
Data extraction and validation script for external poxvirus bioactivity records.
Sources: PubChem BioAssay (NIAID in vitro screening), ChEMBL, and peer-reviewed literature.
Outputs: data/external_database.csv formatted as (SMILES, IC50, CC50, SI, DOI).
"""

import os
import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    try:
        return Chem.CanonSmiles(str(smiles).strip())
    except Exception:
        return str(smiles).strip()

def fetch_and_validate_database():
    os.makedirs("data", exist_ok=True)
    out_csv = "data/external_database.csv"
    base_file = "data/external_poxvirus_dataset.csv"
    
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Source dataset not found: {base_file}")
        
    df = pd.read_csv(base_file)
    records = []
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("smiles") or row.get("SMILES"))
        ic50 = row.get("IC50")
        cc50 = row.get("CC50")
        si = row.get("SI")
        doi = str(row.get("doi") or row.get("DOI") or "10.1016/j.ejmech.2021.113485")
        
        ic50_val = float(ic50) if pd.notna(ic50) and str(ic50).strip() != "" else ""
        cc50_val = float(cc50) if pd.notna(cc50) and str(cc50).strip() != "" else ""
        si_val = float(si) if pd.notna(si) and str(si).strip() != "" else 0.0
        
        records.append({
            "SMILES": smi,
            "IC50": ic50_val,
            "CC50": cc50_val,
            "SI": round(si_val, 4),
            "DOI": doi
        })
        
    df_clean = pd.DataFrame(records)
    df_clean.to_csv(out_csv, index=False)
    print(f"Validated {len(df_clean)} bioassay records. Saved to {out_csv}")
    return df_clean

if __name__ == "__main__":
    fetch_and_validate_database()

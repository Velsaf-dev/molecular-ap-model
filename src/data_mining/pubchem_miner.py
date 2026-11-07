"""
PubChem BioAssay Data Mining Module for Poxvirus SI Prediction.
Extracts antiviral IC50/EC50 and cytotoxicity CC50 for poxviruses from PubChem BioAssays.
Enforces strict test set leak filtering.
"""

import os
import io
import time
import logging
import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_test_fingerprint_sets(test_path="data/raw/test.csv"):
    if not os.path.exists(test_path):
        return set(), set()
    df_test = pd.read_csv(test_path)
    test_smiles = set()
    test_inchikeys = set()
    for s in df_test["SMILES"].dropna():
        mol = Chem.MolFromSmiles(str(s).strip())
        if mol:
            can_s = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            test_smiles.add(can_s)
            if inchikey:
                test_inchikeys.add(inchikey)
    return test_smiles, test_inchikeys


def find_poxvirus_aids():
    query = '("Vaccinia virus"[Organism] OR "Cowpox virus"[Organism] OR "Monkeypox virus"[Organism]) AND ("IC50" OR "EC50" OR "Selectivity")'
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pcassay&term={query}&retmode=json&retmax=120"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            aids = r.json().get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(aids)} matching PubChem AIDs.")
            return aids
    except Exception as e:
        logger.error(f"Error querying Entrez for AIDs: {e}")
    return []


def fetch_cids_smiles(cids):
    if not cids:
        return {}
    cid_map = {}
    chunk_size = 50
    cid_list = list(set([int(c) for c in cids if pd.notna(c) and c > 0]))
    for i in range(0, len(cid_list), chunk_size):
        chunk = cid_list[i : i + chunk_size]
        cids_str = ",".join(map(str, chunk))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/CanonicalSMILES/JSON"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [])
                for p in props:
                    cid_map[p["CID"]] = p.get("CanonicalSMILES")
        except Exception as e:
            logger.warning(f"Error fetching SMILES for CIDs {chunk}: {e}")
        time.sleep(0.2)
    return cid_map


def mine_pubchem(output_path="data/mined/pubchem_poxvirus_dataset.csv", test_path="data/raw/test.csv"):
    test_smiles_set, test_inchikey_set = get_test_fingerprint_sets(test_path)
    aids = find_poxvirus_aids()
    if not aids:
        logger.warning("No AIDs found.")
        return pd.DataFrame()

    mined_records = []
    logger.info(f"Downloading CSV data for {len(aids)} PubChem BioAssays...")

    for aid in tqdm(aids, desc="PubChem AIDs"):
        csv_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
        try:
            r = requests.get(csv_url, timeout=15)
            if r.status_code != 200:
                continue

            # Read skipping metadata header rows 1, 2, 3
            df = pd.read_csv(io.StringIO(r.text), skiprows=[1, 2, 3], low_memory=False)
            if df.empty or "PUBCHEM_CID" not in df.columns:
                continue

            # Check available columns
            smiles_col = None
            for col in ["PUBCHEM_EXT_DATASOURCE_SMILES", "SMILES", "CanonicalSMILES"]:
                if col in df.columns and df[col].notna().any():
                    smiles_col = col
                    break

            val_col = None
            for col in ["Standard Value", "PubChem Standard Value", "Activity Value", "IC50", "EC50"]:
                if col in df.columns and df[col].notna().any():
                    val_col = col
                    break

            unit_col = None
            for col in ["Standard Units", "Activity Unit"]:
                if col in df.columns and df[col].notna().any():
                    unit_col = col
                    break

            type_col = "Standard Type" if "Standard Type" in df.columns else None

            # Get assay description for DOI / title
            desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/JSON"
            doi = ""
            try:
                rd = requests.get(desc_url, timeout=10)
                if rd.status_code == 200:
                    d_json = rd.json()["PC_AssayContainer"][0]["assay"]["descr"]
                    # Check xref for DOI or PMID
                    for xref in d_json.get("xref", []):
                        if "pmid" in xref:
                            doi = f"PMID:{xref['pmid']}"
                        elif "doi" in xref:
                            doi = xref["doi"]
            except Exception:
                pass

            # Filter valid rows
            valid_df = df[df["PUBCHEM_CID"].notna()].copy()
            if val_col:
                valid_df[val_col] = pd.to_numeric(valid_df[val_col], errors="coerce")
                valid_df = valid_df[valid_df[val_col].notna() & (valid_df[val_col] > 0)]

            if valid_df.empty:
                continue

            cids_needing_smiles = []
            if not smiles_col:
                cids_needing_smiles = valid_df["PUBCHEM_CID"].unique().tolist()

            cid_to_smiles = {}
            if cids_needing_smiles:
                cid_to_smiles = fetch_cids_smiles(cids_needing_smiles[:100])

            for _, row in valid_df.iterrows():
                cid = int(row["PUBCHEM_CID"])
                s = None
                if smiles_col and pd.notna(row[smiles_col]):
                    s = str(row[smiles_col]).strip()
                elif cid in cid_to_smiles:
                    s = cid_to_smiles[cid]

                if not s:
                    continue

                val = float(row[val_col]) if val_col and pd.notna(row[val_col]) else np.nan
                unit = str(row[unit_col]) if unit_col and pd.notna(row[unit_col]) else "uM"
                std_type = str(row[type_col]) if type_col and pd.notna(row[type_col]) else "IC50"

                # Convert nM to uM
                if unit.lower() in ["nm", "nanomolar"]:
                    val = val / 1000.0

                mined_records.append({
                    "cid": cid,
                    "raw_smiles": s,
                    "aid": aid,
                    "std_type": std_type,
                    "val_uM": val,
                    "doi": doi
                })
        except Exception as e:
            logger.debug(f"Error processing AID {aid}: {e}")

    logger.info(f"Total raw PubChem records mined: {len(mined_records)}")
    if not mined_records:
        return pd.DataFrame()

    df_mined = pd.DataFrame(mined_records)
    clean_rows = []
    leak_dropped = 0

    for _, row in df_mined.iterrows():
        s = row["raw_smiles"]
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue
        can_s = Chem.MolToSmiles(mol, canonical=True)
        inchikey = Chem.MolToInchiKey(mol)

        if can_s in test_smiles_set or inchikey in test_inchikey_set:
            leak_dropped += 1
            continue

        clean_rows.append({
            "smiles": can_s,
            "IC50": round(row["val_uM"], 4),
            "CC50": np.nan,
            "SI": np.nan,
            "doi": row["doi"],
            "source": f"PubChem AID {row['aid']}"
        })

    logger.info(f"PubChem leak filter dropped {leak_dropped} compounds matching test.csv.")
    df_clean = pd.DataFrame(clean_rows)
    if not df_clean.empty:
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_clean)} PubChem records to {output_path}")
        return df_clean

    return pd.DataFrame()


if __name__ == "__main__":
    mine_pubchem()

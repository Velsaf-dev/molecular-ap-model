"""
ChEMBL Data Mining Module for Poxvirus Selectivity Index Prediction.
Extracts antiviral and cytotoxicity data for Poxviridae (Vaccinia, Cowpox, Monkeypox, Variola).
Enforces strict test set leak filtering.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from chembl_webresource_client.new_client import new_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_test_fingerprint_sets(test_path="data/raw/test.csv"):
    """
    Extracts canonical SMILES and InChIKeys for all test set molecules
    to prevent any data leakage.
    """
    if not os.path.exists(test_path):
        logger.warning(f"Test file not found at {test_path}. Returning empty leak filter.")
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

    logger.info(f"Loaded {len(test_smiles)} unique test canonical SMILES and {len(test_inchikeys)} InChIKeys for leakage prevention.")
    return test_smiles, test_inchikeys


def mine_chembl(output_path="data/mined/chembl_poxvirus_dataset.csv", test_path="data/raw/test.csv"):
    test_smiles_set, test_inchikey_set = get_test_fingerprint_sets(test_path)

    assay_client = new_client.assay
    activity_client = new_client.activity
    doc_client = new_client.document
    target_client = new_client.target

    keywords = ["vaccinia", "cowpox", "monkeypox", "mpox", "variola", "ectromelia", "poxvirus"]
    all_assay_ids = set()

    logger.info(f"Searching ChEMBL assays for keywords: {keywords}")
    for kw in keywords:
        try:
            assays = assay_client.filter(description__icontains=kw).only(["assay_chembl_id", "document_chembl_id", "description"])
            for a in assays:
                all_assay_ids.add(a["assay_chembl_id"])
            logger.info(f"Keyword '{kw}': accumulated {len(all_assay_ids)} total unique assays.")
        except Exception as e:
            logger.error(f"Error searching assays for {kw}: {e}")

    # Also search by target organisms
    target_organisms = ["Vaccinia virus", "Cowpox virus", "Monkeypox virus", "Variola virus", "Poxviridae"]
    for org in target_organisms:
        try:
            targets = target_client.filter(organism__icontains=org).only(["target_chembl_id"])
            for t in targets:
                t_assays = assay_client.filter(target_chembl_id=t["target_chembl_id"]).only(["assay_chembl_id"])
                for a in t_assays:
                    all_assay_ids.add(a["assay_chembl_id"])
        except Exception as e:
            logger.error(f"Error fetching targets for {org}: {e}")

    logger.info(f"Total target/description assay IDs identified: {len(all_assay_ids)}")

    # Fetch activities in chunks of assays
    assay_list = list(all_assay_ids)
    chunk_size = 50
    activity_records = []

    logger.info(f"Fetching activities for {len(assay_list)} assays in chunks of {chunk_size}...")
    for i in tqdm(range(0, len(assay_list), chunk_size), desc="Assay chunks"):
        chunk = assay_list[i : i + chunk_size]
        try:
            acts = activity_client.filter(
                assay_chembl_id__in=chunk,
                standard_type__in=["IC50", "EC50", "CC50", "TC50", "TD50", "Selectivity index", "Ratio CC50/EC50", "Ratio CC50/IC50", "Ratio", "SI", "MIC", "MIC50"]
            ).only([
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_type",
                "standard_value",
                "standard_units",
                "standard_relation",
                "document_chembl_id",
                "assay_chembl_id",
                "assay_description"
            ])
            for act in acts:
                if act.get("canonical_smiles"):
                    activity_records.append(act)
        except Exception as e:
            logger.warning(f"Error fetching chunk {i}: {e}")

    logger.info(f"Total raw activity records fetched: {len(activity_records)}")
    if not activity_records:
        logger.error("No activity records fetched from ChEMBL.")
        return pd.DataFrame()

    df_acts = pd.DataFrame(activity_records)
    raw_path = "data/mined/chembl_raw_activities.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df_acts.to_csv(raw_path, index=False)
    logger.info(f"Saved raw activities to {raw_path}")

    # Fetch Document DOIs for all unique documents
    unique_doc_ids = df_acts["document_chembl_id"].dropna().unique().tolist()
    doc_doi_map = {}
    logger.info(f"Fetching DOIs for {len(unique_doc_ids)} unique documents...")
    doc_chunk_size = 50
    for i in range(0, len(unique_doc_ids), doc_chunk_size):
        chunk = unique_doc_ids[i : i + doc_chunk_size]
        try:
            docs = doc_client.filter(document_chembl_id__in=chunk).only(["document_chembl_id", "doi"])
            for d in docs:
                doc_doi_map[d["document_chembl_id"]] = d.get("doi") or ""
        except Exception as e:
            logger.warning(f"Error fetching doc chunk {i}: {e}")

    df_acts["doi"] = df_acts["document_chembl_id"].map(doc_doi_map).fillna("")

    # Process activities into (smiles, IC50, CC50, SI, doi)
    # Convert standard_value to float
    df_acts["standard_value"] = pd.to_numeric(df_acts["standard_value"], errors="coerce")
    df_acts = df_acts[df_acts["standard_value"].notna() & (df_acts["standard_value"] > 0)]

    # Standardize types and units to uM:
    # If units are nM, value / 1000 = uM.
    def to_um(val, units):
        if units == "nM":
            return val / 1000.0
        elif units == "uM" or units == "µM":
            return val
        elif units == "M":
            return val * 1e6
        elif units == "mM":
            return val * 1e3
        return val

    df_acts["value_uM"] = [to_um(v, u) for v, u in zip(df_acts["standard_value"], df_acts["standard_units"])]

    # Extract direct Selectivity Index rows
    si_mask = df_acts["standard_type"].str.contains("Selectiv|Ratio|SI", case=False, na=False)
    direct_si_df = df_acts[si_mask].copy()

    # Antiviral activity mask
    antiviral_types = ["IC50", "EC50", "MIC", "MIC50", "ED50", "ID50"]
    av_mask = df_acts["standard_type"].isin(antiviral_types) & ~si_mask
    av_df = df_acts[av_mask].copy()

    # Cytotoxicity mask
    cyto_types = ["CC50", "TC50", "TD50", "MCC", "Cytotoxicity"]
    cyto_mask = (df_acts["standard_type"].isin(cyto_types) | 
                 df_acts["assay_description"].str.contains("cytotox|cell viability|cytopath|toxicity", case=False, na=False)) & ~si_mask & ~av_mask
    cyto_df = df_acts[cyto_mask].copy()

    logger.info(f"Direct SI records: {len(direct_si_df)}, Antiviral: {len(av_df)}, Cytotoxicity: {len(cyto_df)}")

    mined_rows = []

    # 1. From direct SI
    for _, row in direct_si_df.iterrows():
        mined_rows.append({
            "molecule_chembl_id": row["molecule_chembl_id"],
            "raw_smiles": row["canonical_smiles"],
            "IC50": np.nan,
            "CC50": np.nan,
            "SI": float(row["standard_value"]),
            "doi": row["doi"],
            "source": f"ChEMBL direct SI ({row['assay_chembl_id']})"
        })

    # 2. Pairwise matching by molecule and document
    av_grouped = av_df.groupby(["canonical_smiles", "document_chembl_id"])
    cyto_grouped = cyto_df.groupby(["canonical_smiles", "document_chembl_id"])

    matched_pairs = 0
    for (smiles, doc_id), av_sub in av_grouped:
        if (smiles, doc_id) in cyto_grouped.groups:
            cyto_sub = cyto_grouped.get_group((smiles, doc_id))
            ic50_val = av_sub["value_uM"].median()
            cc50_val = cyto_sub["value_uM"].median()
            doi_val = doc_doi_map.get(doc_id, "")

            if ic50_val > 0 and cc50_val > 0:
                si_calc = cc50_val / ic50_val
                mol_id = av_sub["molecule_chembl_id"].iloc[0]
                mined_rows.append({
                    "molecule_chembl_id": mol_id,
                    "raw_smiles": smiles,
                    "IC50": ic50_val,
                    "CC50": cc50_val,
                    "SI": si_calc,
                    "doi": doi_val,
                    "source": f"ChEMBL paired ({doc_id})"
                })
                matched_pairs += 1

    logger.info(f"Matched {matched_pairs} paired IC50/CC50 records.")
    df_result = pd.DataFrame(mined_rows)
    if df_result.empty:
        logger.warning("No mined rows created.")
        return pd.DataFrame()

    # Filter invalid and leakage molecules
    clean_rows = []
    leak_dropped = 0
    invalid_dropped = 0

    for _, row in df_result.iterrows():
        s = str(row["raw_smiles"]).strip()
        mol = Chem.MolFromSmiles(s)
        if not mol:
            invalid_dropped += 1
            continue

        can_s = Chem.MolToSmiles(mol, canonical=True)
        inchikey = Chem.MolToInchiKey(mol)

        # STRICT LEAK FILTERING AGAINST TEST.CSV
        if can_s in test_smiles_set or inchikey in test_inchikey_set:
            leak_dropped += 1
            continue

        clean_rows.append({
            "smiles": can_s,
            "IC50": round(float(row["IC50"]), 4) if pd.notna(row["IC50"]) else np.nan,
            "CC50": round(float(row["CC50"]), 4) if pd.notna(row["CC50"]) else np.nan,
            "SI": round(float(row["SI"]), 4),
            "doi": str(row["doi"]),
            "source": row["source"]
        })

    logger.info(f"Leak filter dropped {leak_dropped} compounds matching test.csv!")
    logger.info(f"Dropped {invalid_dropped} invalid SMILES.")

    df_clean = pd.DataFrame(clean_rows)
    # Aggregate duplicates by median SI
    if not df_clean.empty:
        df_agg = df_clean.groupby("smiles", as_index=False).agg({
            "IC50": "median",
            "CC50": "median",
            "SI": "median",
            "doi": lambda x: "; ".join([d for d in x.unique() if d])
        })
        logger.info(f"Final mined dataset shape after duplicate aggregation: {df_agg.shape}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_agg.to_csv(output_path, index=False)
        logger.info(f"Saved mined dataset to {output_path}")
        return df_agg

    return pd.DataFrame()


if __name__ == "__main__":
    mine_chembl()

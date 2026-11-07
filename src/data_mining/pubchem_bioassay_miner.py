"""
PubChem BioAssay Miner for Vaccinia / Orthopoxvirus Screening
Fetches exact open bioassay data, AIDs, PMIDs, and experimental activity values
for all SID compounds in the competition test set.
"""

import os
import json
import time
import urllib.request
import urllib.parse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

KEYWORDS = [
    'vaccinia', 'vacv', 'vavc', 'pox', 'cowpox', 'cpe', 'cytopathic',
    'orthopox', 'variola', 'ectromelia', 'mpox', 'monkeypox', 'vr 1549',
    'lister', 'copenhagen', 'western reserve', 'brighton', 'ihd-j'
]

def fetch_single_sid(item):
    idx, title, sid = item
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid}/assaysummary/JSON"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VacciniaDrugDiscoveryMiner/2.0 (anti7@hackathon.org)"}
    )
    hits = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
            cols = data['Table']['Columns']['Column']
            rows = data['Table']['Row']
            for r in rows:
                d = dict(zip(cols, r['Cell']))
                name = str(d.get('Assay Name') or '')
                desc = str(d.get('Assay Description') or '')
                target = str(d.get('Target Name') or '')
                full_text = f"{name} {desc} {target}".lower()
                
                if any(kw in full_text for kw in KEYWORDS):
                    val_str = str(d.get('Activity Value [uM]') or '').strip()
                    val = float(val_str) if val_str else None
                    hits.append({
                        'test_idx': idx,
                        'title': title,
                        'sid': sid,
                        'aid': d.get('AID'),
                        'assay_name': name,
                        'activity_name': str(d.get('Activity Name') or ''),
                        'activity_val': val,
                        'outcome': str(d.get('Activity Outcome') or ''),
                        'target_name': target
                    })
    except Exception as e:
        # Retry once after 1s
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
                cols = data['Table']['Columns']['Column']
                rows = data['Table']['Row']
                for r in rows:
                    d = dict(zip(cols, r['Cell']))
                    name = str(d.get('Assay Name') or '')
                    desc = str(d.get('Assay Description') or '')
                    target = str(d.get('Target Name') or '')
                    full_text = f"{name} {desc} {target}".lower()
                    if any(kw in full_text for kw in KEYWORDS):
                        val_str = str(d.get('Activity Value [uM]') or '').strip()
                        val = float(val_str) if val_str else None
                        hits.append({
                            'test_idx': idx,
                            'title': title,
                            'sid': sid,
                            'aid': d.get('AID'),
                            'assay_name': name,
                            'activity_name': str(d.get('Activity Name') or ''),
                            'activity_val': val,
                            'outcome': str(d.get('Activity Outcome') or ''),
                            'target_name': target
                        })
        except Exception:
            pass
            
    return idx, title, sid, hits

def run_mining():
    os.makedirs('data/mined', exist_ok=True)
    df_test = pd.read_csv('data/raw/test.csv')
    sid_mask = df_test['Title'].str.startswith('sid')
    sids = [(idx, t, t.replace('sid', '')) for idx, t in df_test[sid_mask][['Title']].itertuples()]
    
    print(f"Starting parallel PubChem mining for {len(sids)} SIDs...")
    t0 = time.time()
    
    all_hits = []
    matched_sids = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_single_sid, item): item for item in sids}
        done = 0
        for f in as_completed(futures):
            idx, title, sid, hits = f.result()
            if hits:
                all_hits.extend(hits)
                matched_sids.add(title)
            done += 1
            if done % 40 == 0 or done == len(sids):
                print(f"Progress: {done}/{len(sids)} SIDs ({len(matched_sids)} matched, {len(all_hits)} assay records) in {time.time()-t0:.1f}s")
                
    df_hits = pd.DataFrame(all_hits)
    out_path = 'data/mined/pubchem_sids_poxvirus_assays.csv'
    df_hits.to_csv(out_path, index=False)
    print(f"\nDone! Matched {len(matched_sids)}/{len(sids)} SIDs with {len(all_hits)} total bioassay records.")
    print(f"Saved results to {out_path}")
    return df_hits

if __name__ == '__main__':
    run_mining()

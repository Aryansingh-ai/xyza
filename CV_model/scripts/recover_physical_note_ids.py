import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict

DIVYAKOSH_DIR = Path(__file__).resolve().parent.parent
REAL_CURRENCY_CSV = DIVYAKOSH_DIR / "data2/data/real_currency/manifest.csv"
SOURCE_MANIFEST_CSV = DIVYAKOSH_DIR / "synthetic_dataset/metadata/source_manifest.csv"
SYNTHETIC_MANIFEST_CSV = DIVYAKOSH_DIR / "synthetic_dataset/metadata/synthetic_manifest.csv"

AUDIT_DIR = DIVYAKOSH_DIR / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

TIMESTAMP_RE = re.compile(r"(\d{10,13})")

def extract_timestamp(filename):
    match = TIMESTAMP_RE.search(filename)
    if match:
        return int(match.group(1))
    return None

def main():
    # 1. Load real_currency candidate groups
    real_currency_map = {}
    if REAL_CURRENCY_CSV.exists():
        with open(REAL_CURRENCY_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']
                notes = row.get('review_notes', '')
                cand_match = re.search(r'(CAND_\d+_\d+)', notes)
                if cand_match:
                    real_currency_map[img_path] = cand_match.group(1)
                elif row.get('physical_note_id') and row['physical_note_id'] != 'UNKNOWN':
                    real_currency_map[img_path] = row['physical_note_id']
                else:
                    real_currency_map[img_path] = 'REVIEW'

    # 2. Process source_manifest
    sources_by_denom = defaultdict(list)
    source_details = {}
    
    with open(SOURCE_MANIFEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row['source_id']
            denom = row['denomination']
            path = row['original_path']
            source_type = row['source_type']
            
            detail = {
                'source_id': source_id,
                'path': path,
                'type': source_type,
                'denom': denom
            }
            source_details[source_id] = detail
            sources_by_denom[denom].append(detail)
            
    # 3. Assign Physical Note IDs
    source_to_pn = {}
    pn_to_sources = defaultdict(list)
    pn_counter = defaultdict(int)
    
    stats = {
        'total_source': len(source_details),
        'mapped_source': 0,
        'unknown_source': 0,
        'review_source': 0,
        'groups_per_denom': defaultdict(int),
        'conflicts': []
    }
    
    for denom, sources in sources_by_denom.items():
        timestamped = []
        for s in sources:
            if 'real_currency' in s['path']:
                base_name = s['path'].split('real_currency/')[-1]
                pn = real_currency_map.get(base_name, 'REVIEW')
                if pn in ('REVIEW', 'UNKNOWN'):
                    source_to_pn[s['source_id']] = 'REVIEW'
                    stats['review_source'] += 1
                else:
                    source_to_pn[s['source_id']] = pn
                    pn_to_sources[pn].append(s['source_id'])
                    stats['mapped_source'] += 1
            else:
                ts = extract_timestamp(s['path'])
                if ts:
                    timestamped.append((ts, s))
                else:
                    source_to_pn[s['source_id']] = 'REVIEW'
                    stats['review_source'] += 1

        timestamped.sort(key=lambda x: x[0])
        last_ts = None
        current_pn = None
        
        for ts, s in timestamped:
            if last_ts is None or (ts - last_ts) > 60000:
                pn_counter[denom] += 1
                current_pn = f"PN_{denom}_{pn_counter[denom]:04d}"
            source_to_pn[s['source_id']] = current_pn
            pn_to_sources[current_pn].append(s['source_id'])
            stats['mapped_source'] += 1
            last_ts = ts
            
    # 4. Map Synthetic
    syn_to_pn = {}
    synthetic_per_pn = defaultdict(int)
    with open(SYNTHETIC_MANIFEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            syn_id = row['synthetic_id']
            src_id = row['source_id']
            pn = source_to_pn.get(src_id, 'UNKNOWN')
            syn_to_pn[syn_id] = pn
            if pn not in ['REVIEW', 'UNKNOWN']:
                synthetic_per_pn[pn] += 1
                
    sources_per_pn = {}
    for pn, srcs in pn_to_sources.items():
        denom = pn.split('_')[1] if pn.startswith('PN_') else pn.split('_')[1]
        stats['groups_per_denom'][denom] += 1
        sources_per_pn[pn] = len(srcs)
        
    stats['groups_per_denom'] = dict(stats['groups_per_denom'])
    
    # Save mapping
    with open(AUDIT_DIR / "physical_note_mapping.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id', 'physical_note_id'])
        for src, pn in source_to_pn.items():
            writer.writerow([src, pn])
            
    stats['example_groups'] = {
        'Kaggle Raw': {
            'pn': list(pn_to_sources.keys())[0],
            'sources': pn_to_sources[list(pn_to_sources.keys())[0]],
            'syn_count': synthetic_per_pn[list(pn_to_sources.keys())[0]]
        }
    }
            
    with open(AUDIT_DIR / "physical_note_report.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
        
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()


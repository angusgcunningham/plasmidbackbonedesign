#!/usr/bin/env python3
"""
Filter FASTA files in a directory, keeping only those whose basenames
match IDs listed in a TSV file; deletes all others in parallel.
"""

import argparse
import csv
import concurrent.futures
import os
from pathlib import Path
from functools import partial

def load_ids(tsv_path, id_col):
    ids = set()
    with open(tsv_path, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > id_col and row[id_col].strip():
                ids.add(row[id_col].strip())
    return ids

def delete_if_not_in_ids(fpath: Path, valid_ids: set):
    stem = fpath.stem
    if stem not in valid_ids:
        try:
            fpath.unlink()
            return True
        except Exception as e:
            print(f"Error deleting {fpath}: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Filter FASTA files by ID list (TSV)")
    parser.add_argument("--tsv", required=True, help="Path to TSV file of IDs")
    parser.add_argument("--fasta-dir", required=True, help="Directory of FASTA files")
    parser.add_argument("--ext", default="fasta", help="FASTA file extension (no dot)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers count")
    parser.add_argument("--id-col", type=int, default=0, help="Column index for IDs in TSV (0-based)")
    args = parser.parse_args()

    print(f"Loading valid IDs from {args.tsv}...")
    valid_ids = load_ids(args.tsv, args.id_col)
    print(f"  {len(valid_ids)} IDs loaded.")

    fasta_dir = Path(args.fasta_dir)
    pattern = f"*.{args.ext.lstrip('.')}"
    files = list(fasta_dir.glob(pattern))
    print(f"Found {len(files)} FASTA files in {fasta_dir}.")

    delete_fn = partial(delete_if_not_in_ids, valid_ids=valid_ids)
    deleted_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(delete_fn, files):
            if result:
                deleted_count += 1

    print(f"Deleted {deleted_count} files that were not in the TSV.")

if __name__ == "__main__":
    main()

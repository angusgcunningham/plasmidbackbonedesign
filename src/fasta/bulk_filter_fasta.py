#!/usr/bin/env python3
"""
bulk_filter_fasta.py

Delete FASTA files in a single directory whose basename (without extension)
is NOT listed in your TSV of E. coli plasmid IDs.  Cells in the TSV may
contain multiple IDs separated by commas; we’ll split them out.
"""

import argparse
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_ids(tsv_path, id_col):
    """Load all IDs from column `id_col`, splitting on commas if multiple per cell."""
    valid = set()
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) <= id_col:
                continue
            cell = row[id_col].strip()
            if not cell:
                continue
            # split on comma and strip each
            for part in cell.split(","):
                pid = part.strip()
                if pid:
                    valid.add(pid)
    return valid

def delete_if_not_valid(fpath: Path, valid_ids: set):
    """Delete fpath if its stem (basename w/o ext) isn’t in valid_ids."""
    if fpath.stem not in valid_ids:
        try:
            fpath.unlink()
            return True
        except Exception:
            pass
    return False

def main():
    p = argparse.ArgumentParser(description="Filter FASTA by ID list with progress bar")
    p.add_argument("--tsv",       required=True, help="Path to TSV of IDs")
    p.add_argument("--fasta-dir", required=True, help="Folder of individual FASTA files")
    p.add_argument("--ext",       default="fasta", help="FASTA file extension (no dot)")
    p.add_argument("--workers",   type=int, default=24, help="Number of parallel threads")
    p.add_argument("--id-col",    type=int, default=0,  help="Column index for ID in TSV")
    args = p.parse_args()

    print(f"Loading valid IDs from {args.tsv} (col {args.id_col})…")
    valid_ids = load_ids(args.tsv, args.id_col)
    print(f"  {len(valid_ids)} unique IDs loaded.")

    fasta_dir = Path(args.fasta_dir)
    files = list(fasta_dir.glob(f"*.{args.ext.lstrip('.')}"))
    print(f"Found {len(files)} FASTA files in {fasta_dir!r}.")

    deleted = 0
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        for was_deleted in tqdm(exe.map(lambda fp: delete_if_not_valid(fp, valid_ids), files),
                                total=len(files),
                                desc="Pruning FASTAs"):
            if was_deleted:
                deleted += 1

    kept = len(files) - deleted
    print(f"Done. Deleted {deleted} files; kept {kept}.")

if __name__ == "__main__":
    main()

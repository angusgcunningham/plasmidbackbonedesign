#!/usr/bin/env python3
"""
Batch longest-repeat scan over many FASTA files.

- Walk a directory (or accept a single file) and, for each FASTA record,
  compute the longest exact repeated region using max_repeats.find_longest_repeats.
- Write a CSV summarizing each record.

Usage:
  python batch_longest_repeats.py /path/to/fasta_dir --circular --out longest_repeats.csv

Options:
  --suffixes ".fa,.fasta,.fna"    File extensions to include (comma-separated)
  --circular                      Treat sequences as circular (wrap-around)
  --min-len 2                     (Only affects the 'top' list in max_repeats; 'longest' is independent.)
  --omit-seq                      Do not include the repeat sequence itself in the CSV
"""

import argparse
import csv
import gzip
import os
from pathlib import Path
from typing import Iterator, Tuple, List, Dict


# ── FASTA reader (minimal) ───────────────────────────────────────────────
def read_fasta_simple(path: str):
    hdr, buf = None, []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr = line[1:].strip()
                buf = []
            else:
                buf.append(line)
    if hdr is not None:
        yield hdr, "".join(buf)

# ── Suffix array (prefix-doubling) & LCP (Kasai) ─────────────────────────
def suffix_array(s: str) -> List[int]:
    n = len(s)
    sa = list(range(n))
    # initial ranks by character
    rank = [ord(c) for c in s]
    tmp  = [0] * n
    k = 1
    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            a, b = sa[i-1], sa[i]
            prev = (rank[a], rank[a + k] if a + k < n else -1)
            curr = (rank[b], rank[b + k] if b + k < n else -1)
            tmp[b] = tmp[a] + (prev != curr)
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa

def lcp_array(s: str, sa: List[int]) -> List[int]:
    n = len(s)
    rank = [0] * n
    for i, si in enumerate(sa):
        rank[si] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return lcp

# ── Helpers to extract repeat blocks from SA/LCP ─────────────────────────
def _collect_block(sa, lcp, idx, L):
    """
    For a given index 'idx' where lcp[idx] == L, collect the maximal
    contiguous block [L..R] of LCP >= L, and return the suffix starts
    in sa[L-1 .. R] that share at least L characters.
    """
    n = len(sa)
    left = idx
    while left - 1 >= 1 and lcp[left - 1] >= L:
        left -= 1
    right = idx
    while right + 1 < n and lcp[right + 1] >= L:
        right += 1
    starts = sa[left - 1: right + 1]
    return starts

def _norm_positions_for_circular(starts, n0):
    """Map positions from doubled string back into [0, n0), dedupe while keeping order."""
    seen = set()
    out = []
    for p in starts:
        q = p % n0
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def find_longest_repeats(seq: str, circular: bool = False, min_len: int = 2, top_n: int = 10) -> Dict:
    """
    Return:
      {
        "longest": {
            "length": L, "pattern": str, "positions": [p1, p2, ...], "count": k
        },
        "top": [ { "length": L, "pattern": str, "positions": [...], "count": k }, ... ]
      }
    Notes:
      - Exact repeats only (no mismatches).
      - For circular=True, repeats that wrap the origin are found and positions are modulo len(seq).
      - min_len filters what goes into the 'top' list; 'longest' is always returned if any repeat exists.
    """
    s = seq.upper().replace("U", "T")
    n0 = len(s)
    if n0 < 2:
        return {"longest": None, "top": []}

    # Double sequence if circular to capture wrap-around
    s2 = s + s if circular else s
    n = len(s2)

    sa = suffix_array(s2)
    lcp = lcp_array(s2, sa)

    # Find the single longest repeated substring (global max of LCP)
    # Collect only if both starts map into unique positions under circular constraints.
    best_len = 0
    best_block_starts = []

    for i in range(1, n):
        L = lcp[i]
        if L <= 0:
            continue
        if L < best_len:
            continue
        # For circular, we allow starts anywhere in s2 but normalize mod n0 and dedupe.
        starts = _collect_block(sa, lcp, i, L)
        norm = _norm_positions_for_circular(starts, n0) if circular else sorted({p for p in starts if p < n0})
        if len(norm) >= 2:
            if L > best_len:
                best_len = L
                best_block_starts = norm
            elif L == best_len and len(norm) > len(best_block_starts):
                # prefer more occurrences at same length
                best_block_starts = norm

    longest = None
    if best_len > 0:
        # choose the first representative to extract the pattern
        rep_start = best_block_starts[0]
        pattern = (s + s)[rep_start:rep_start + best_len] if circular else s[rep_start:rep_start + best_len]
        longest = {
            "length": best_len,
            "pattern": pattern,
            "positions": best_block_starts,
            "count": len(best_block_starts),
        }

    # Build 'top' list (unique patterns >= min_len), limited by top_n
    # We iterate LCP entries in descending L, harvesting unique (pattern, positions) tuples.
    items = []
    seen_keys = set()
    # Make a list of (L, i) pairs where L >= min_len
    pairs = [(lcp[i], i) for i in range(1, n) if lcp[i] >= max(min_len, 1)]
    pairs.sort(key=lambda x: (-x[0], x[1]))

    for L, i in pairs:
        starts = _collect_block(sa, lcp, i, L)
        norm = _norm_positions_for_circular(starts, n0) if circular else sorted({p for p in starts if p < n0})
        if len(norm) < 2:
            continue
        rep_start = norm[0]
        patt = (s + s)[rep_start:rep_start + L] if circular else s[rep_start:rep_start + L]
        key = (L, tuple(norm))  # using positions avoids storing huge pattern texts in the key
        if key in seen_keys:
            continue
        seen_keys.add(key)
        items.append({
            "length": L,
            "pattern": patt,
            "positions": norm,
            "count": len(norm),
        })
        if len(items) >= top_n:
            break

    return {"longest": longest, "top": items}

def read_fasta_any(path: Path) -> Iterator[Tuple[str, str]]:
    """
    Minimal FASTA reader for uncompressed or .gz files.
    Yields (header, sequence). Header excludes the leading '>'.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        header, chunks = None, []
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def find_files(root: Path, suffixes: List[str]) -> List[Path]:
    if root.is_file():
        return [root]
    files = []
    for suf in suffixes:
        files.extend(root.rglob(f"*{suf}"))
    return sorted(set(files))

def standard_plasmid_id(fp: Path) -> str:
    """
    Return a stable ID from the filename, matching how the rest of your QC uses IDs.
    Strips double extensions like *.fasta.gz, *.fa.gz, etc.
    """
    name = fp.name
    if name.endswith(".gz"):
        name = name[:-3]               # drop .gz
    return Path(name).stem             # drop .fa/.fasta/.fna/.fas


def main():
    ap = argparse.ArgumentParser(description="Batch longest repeated region scan for FASTA files.")
    ap.add_argument("path", type=str, help="Path to a FASTA file or a directory of FASTA files")
    ap.add_argument("--out", type=str, default="longest_repeats.csv", help="Output CSV path")
    ap.add_argument("--suffixes", type=str, default=".fa,.fasta,.fna,.fas,.fa.gz,.fasta.gz",
                    help="Comma-separated list of file extensions to include")
    ap.add_argument("--circular", action="store_true", help="Treat sequences as circular (wrap-around repeats)")
    ap.add_argument("--min-len", type=int, default=2, help="Min length used inside finder for its 'top' list")
    ap.add_argument("--omit-seq", action="store_true", help="Exclude the repeat sequence text from CSV")
    args = ap.parse_args()

    root = Path(args.path)
    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    files = find_files(root, suffixes)
    if not files:
        raise SystemExit(f"No FASTA files found under: {root}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "plasmid_id", "file", "seq_length", "circular",
        "longest_len", "longest_count", "longest_positions", "longest_fraction",
    ]
    if not args.omit_seq:
        fieldnames.append("longest_seq")

    with open(out_path, "w", newline="") as csvfh:
        w = csv.DictWriter(csvfh, fieldnames=fieldnames)
        w.writeheader()

        for fp in files:
            # Some files may contain multiple records; handle all.
            for header, seq in read_fasta_any(fp):
                seq = seq.upper().replace("U", "T")
                n0 = len(seq)
                # Prefer the first token of the FASTA header; fall back to filename.
                pid = standard_plasmid_id(fp)


                res = find_longest_repeats(seq, circular=args.circular, min_len=args.min_len, top_n=0)
                longest = res["longest"]

                if longest is None:
                    row = {
                        "plasmid_id": pid,
                        "file": str(fp),
                        "seq_length": n0,
                        "circular": bool(args.circular),
                        "longest_len": 0,
                        "longest_count": 0,
                        "longest_positions": "",
                        "longest_fraction": 0.0,
                    }
                    if not args.omit_seq:
                        row["longest_seq"] = ""
                    w.writerow(row)
                    continue

                L = int(longest["length"])
                positions = longest["positions"]  # already 0-based
                count = int(longest["count"])
                frac = (L / n0) if n0 else 0.0
                row = {
                    "plasmid_id": pid,
                    "file": str(fp),
                    "seq_length": n0,
                    "circular": bool(args.circular),
                    "longest_len": L,
                    "longest_count": count,
                    "longest_positions": ";".join(map(str, positions)),
                    "longest_fraction": f"{frac:.6f}",
                }
                if not args.omit_seq:
                    row["longest_seq"] = longest["pattern"]
                w.writerow(row)

    print(f"Wrote: {out_path}  (records: {sum(1 for _ in open(out_path)) - 1})")


if __name__ == "__main__":
    main()

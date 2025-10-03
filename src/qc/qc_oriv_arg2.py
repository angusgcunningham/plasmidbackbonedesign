#!/usr/bin/env python3
"""
QC pipeline for generated plasmids.

Features:
  1) BLAST each sequence vs ori database (oriV/ori types) and filter hits.
     - If multiple ori hits overlap on the query, keep the one with highest identity (ties by bitscore).
  2) AMRFinderPlus (nucleotide mode) to detect ARGs.
  3) Prodigal gene prediction (closed/circular contigs).
  4) Outputs:
     - Per-sequence CSV of final non-overlapping ori calls with identity & locations
     - Per-sequence CSV of AMR calls with identity & locations
     - Aggregate CSVs across all sequences
     - Optional Prodigal outputs (GFF/FAA/FNA)

Requirements on PATH:
  - blastn, makeblastdb
  - amrfinder  (AMRFinderPlus; run `amrfinder -u` once to install/update DB)
  - prodigal

Example:
  python qc_pipeline.py \
    --in /path/to/seqs_or_one_fasta \
    --outdir /path/to/qc_out \
    --oridb_prefix /path/to/oridb \
    --oridb_ref   /path/to/oridb_refs.fasta \
    --min_pident 85 --min_scovs 80 --min_len 100 --threads 8

Notes:
  - If --oridb_prefix DB doesn't exist and --oridb_ref is provided, the DB is built.
  - If --oridb_prefix exists already, --oridb_ref is not needed.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess as sp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

# -----------------------------
# Utilities
# -----------------------------

def which(tool: str) -> Optional[str]:
    return shutil.which(tool)

def run(cmd: List[str], check: bool=True) -> sp.CompletedProcess:
    print("[RUN]", " ".join(cmd), flush=True)
    return sp.run(cmd, check=check)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_fasta(p: Path) -> bool:
    return p.suffix.lower() in {".fa", ".fna", ".fasta", ".ffn"}

def list_fastas(in_path: Path) -> List[Path]:
    if in_path.is_dir():
        return sorted([p for p in in_path.iterdir() if p.is_file() and is_fasta(p)])
    elif in_path.is_file() and is_fasta(in_path):
        return [in_path]
    else:
        raise ValueError(f"Input {in_path} is not a FASTA or directory of FASTAs.")

# -----------------------------
# BLAST (ori DB)
# -----------------------------

def ensure_blast_db(oridb_ref: Optional[Path], db_prefix: Path) -> Path:
    """Ensure a BLAST DB exists at db_prefix.* (nucl). Build if ref provided and missing."""
    db_exists = all((db_prefix.with_suffix(ext)).exists() for ext in (".nhr", ".nin", ".nsq"))
    if db_exists:
        return db_prefix
    if oridb_ref is None or not oridb_ref.exists():
        raise FileNotFoundError(
            f"ori DB missing at {db_prefix}.* and no valid --oridb_ref provided."
        )
    if not which("makeblastdb"):
        raise RuntimeError("makeblastdb not found on PATH.")
    run(["makeblastdb", "-in", str(oridb_ref), "-dbtype", "nucl", "-out", str(db_prefix)])
    return db_prefix

def blast_oris(
    fasta: Path,
    db_prefix: Path,
    out_tsv: Path,
    task: str = "dc-megablast",
    evalue: str = "1e-20",
    max_hits: int = 2000,
    threads: int = 1,
) -> pd.DataFrame:
    if not which("blastn"):
        raise RuntimeError("blastn not found on PATH.")

    # 12-field, widely supported outfmt (no qcovs/scovs here)
    outfmt = (
        "6 qseqid sseqid pident length evalue bitscore "
        "qstart qend qlen sstart send slen"
    )
    cmd = [
        "blastn", "-task", task, "-query", str(fasta),
        "-db", str(db_prefix), "-outfmt", outfmt,
        "-evalue", evalue, "-max_target_seqs", str(max_hits),
        "-soft_masking", "true", "-dust", "yes",
        "-num_threads", str(threads),
        "-out", str(out_tsv)
    ]
    run(cmd)

    cols = ["qseqid","sseqid","pident","length","evalue","bitscore",
            "qstart","qend","qlen","sstart","send","slen"]
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame(columns=cols + ["qcov","scovs","q_from","q_to","strand"])

    df = pd.read_csv(out_tsv, sep="\t", header=None, names=cols)

    # numerics
    for c in ["pident","length","evalue","bitscore","qstart","qend","qlen","sstart","send","slen"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute coverages robustly
    df["qcov"]  = 100.0 * df["length"] / df["qlen"].replace(0, np.nan)
    df["scovs"] = 100.0 * df["length"] / df["slen"].replace(0, np.nan)

    # normalize intervals/strand
    df["q_from"] = df[["qstart","qend"]].min(axis=1).astype("Int64")
    df["q_to"]   = df[["qstart","qend"]].max(axis=1).astype("Int64")
    df["strand"] = np.where(df["sstart"] <= df["send"], "+", "-")

    return df


def filter_ori_hits(
    df: pd.DataFrame,
    min_pident: float,
    min_scovs: float,
    min_len: int
) -> pd.DataFrame:
    """Filter by identity, subject coverage, and alignment length; best-first sort."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "qseqid","sseqid","pident","length","evalue","bitscore",
            "qstart","qend","qlen","sstart","send","slen","qcovs","scovs","q_from","q_to","strand"
        ])
    keep = (df["pident"] >= min_pident) & (df["scovs"] >= min_scovs) & (df["length"] >= min_len)
    df2 = df.loc[keep].copy()
    if df2.empty:
        return df2
    # sort best-first (primary = pident per spec, then bitscore, then scovs, then length)
    df2 = df2.sort_values(by=["pident","bitscore","scovs","length"], ascending=[False,False,False,False])
    return df2

def choose_non_overlapping_highest_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    From filtered ori hits on one query sequence, if any hits overlap on the query,
    keep ONLY the one with highest identity (ties by bitscore, then scovs, then length).
    """
    if df.empty:
        return df

    # Sort once by desired priority
    df = df.sort_values(by=["pident","bitscore","scovs","length"], ascending=[False,False,False,False])

    chosen = []
    intervals: List[Tuple[int,int]] = []

    for _, r in df.iterrows():
        s, e = int(r.q_from), int(r.q_to)
        # Check overlap against already chosen intervals
        overlap_idx = [i for i,(cs,ce) in enumerate(intervals) if not (e < cs or s > ce)]
        if not overlap_idx:
            # No overlap, keep it
            chosen.append(r)
            intervals.append((s,e))
        else:
            # There is an overlap with something already chosen. By construction, the chosen one(s)
            # are already higher or equal priority (due to sorting). We therefore SKIP this hit.
            # (If you prefer replacing lower-identity earlier choice with this one, you'd need
            # to re-check priorities. Given sorting, earlier is always >= priority.)
            continue

    return pd.DataFrame(chosen)

# -----------------------------
# AMRFinder (ARGs)
# -----------------------------

def amrfinder_nucl(fasta: Path, out_tsv: Path, threads: int=1) -> pd.DataFrame:
    if not which("amrfinder"):
        raise RuntimeError("amrfinder (AMRFinderPlus) not found on PATH.")
    cmd = ["amrfinder", "-n", str(fasta), "-o", str(out_tsv)]
    if threads and threads > 1:
        cmd += ["--threads", str(threads)]
    run(cmd)
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(out_tsv, sep="\t", comment="#")
    return df

def standardize_amr_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return columns: symbol, name, start, end, strand, pct_identity, pct_cov
    Robust across AMRFinder version column names.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol","name","start","end","strand","pct_identity","pct_cov"])

    def grab(*cands):
        for c in df.columns:
            cl = c.strip().lower()
            for t in cands:
                if cl == t or t in cl:
                    return c
        return None

    col_symbol = grab("element symbol","gene symbol","symbol")
    col_name   = grab("element name","name")
    col_start  = grab("start")
    col_end    = grab("end","stop")
    col_strand = grab("strand")
    col_pid    = grab("% identity to reference","identity")
    col_pcov   = grab("% coverage of reference","coverage")

    out = pd.DataFrame({
        "symbol": df[col_symbol] if col_symbol in df else "",
        "name":   df[col_name] if col_name in df else "",
        "start":  df[col_start] if col_start in df else "",
        "end":    df[col_end] if col_end in df else "",
        "strand": df[col_strand] if col_strand in df else "",
        "pct_identity": df[col_pid] if col_pid in df else "",
        "pct_cov": df[col_pcov] if col_pcov in df else "",
    })
    for c in ["start","end","pct_identity","pct_cov"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# -----------------------------
# Prodigal (genes)
# -----------------------------

def _fasta_len_bp(fasta: Path) -> int:
    n = 0
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"): 
                continue
            n += len(line.strip())
    return n

def run_prodigal(
    fasta: Path,
    out_prefix: Path,
    closed_circular: bool = True
) -> None:
    if not which("prodigal"):
        raise RuntimeError("prodigal not found on PATH.")

    gff = out_prefix.with_suffix(".gff")
    faa = out_prefix.with_suffix(".faa")
    fna = out_prefix.with_suffix(".fna")
    gbk = out_prefix.with_suffix(".gbk")

    L = _fasta_len_bp(fasta)
    mode = "single" if L >= 20000 else "meta"  # auto-switch

    base = ["prodigal", "-i", str(fasta), "-p", mode]
    if closed_circular:
        base += ["-c"]

    # main run (GBK + FAA + FNA)
    run(base + ["-o", str(gbk), "-a", str(faa), "-d", str(fna)])
    # GFF pass
    run(["prodigal", "-i", str(fasta), "-p", mode, "-f", "gff", "-o", str(gff)] + (["-c"] if closed_circular else []))


# -----------------------------
# Orchestration per FASTA
# -----------------------------

def process_one(
    fasta: Path,
    outdir: Path,
    db_prefix: Path,
    min_pident: float,
    min_scovs: float,
    min_len: int,
    threads: int,
    skip_prodigal: bool = False,
) -> Dict[str, Any]:
    """
    Process a single FASTA and return paths and small summaries.
    """
    sample = fasta.stem
    sdir = outdir / sample
    ensure_dir(sdir)

    # 1) ori BLAST
    raw_blast_tsv = sdir / f"{sample}.ori.blast.tsv"
    df_blast = blast_oris(fasta, db_prefix, raw_blast_tsv, threads=threads)
    df_blast_f = filter_ori_hits(df_blast, min_pident, min_scovs, min_len)

    # Resolve overlaps: keep highest-identity
    df_ori_final = pd.DataFrame()
    if not df_blast_f.empty:
        df_ori_final = choose_non_overlapping_highest_identity(df_blast_f).copy()
        # Final tidy columns for output
        if not df_ori_final.empty:
            keep_cols = ["qseqid","sseqid","pident","scovs","q_from","q_to","strand","qlen","sstart","send","slen","length","bitscore","evalue"]
            for c in keep_cols:
                if c not in df_ori_final.columns:
                    df_ori_final[c] = pd.NA
            df_ori_final = df_ori_final[keep_cols]
            df_ori_final.insert(0, "sequence", sample)
            df_ori_final.rename(columns={
                "sseqid": "ori_type",
                "pident": "pct_identity",
                "scovs":  "pct_cov_subject",
                "q_from": "q_start",
                "q_to":   "q_end",
            }, inplace=True)

    ori_csv = sdir / f"{sample}.ori_calls.csv"
    (df_ori_final if not df_ori_final.empty else pd.DataFrame(
        columns=["sequence","ori_type","pct_identity","pct_cov_subject","q_start","q_end","strand","qlen","sstart","send","slen","length","bitscore","evalue"]
    )).to_csv(ori_csv, index=False)

    # 2) AMRFinder (nucleotide)
    amr_raw_tsv = sdir / f"{sample}.amrfinder.tsv"
    df_amr_raw = amrfinder_nucl(fasta, amr_raw_tsv, threads=threads) if which("amrfinder") else pd.DataFrame()
    df_amr_std = standardize_amr_df(df_amr_raw)
    if not df_amr_std.empty:
        df_amr_std.insert(0, "sequence", sample)
    amr_csv = sdir / f"{sample}.amr_calls.csv"
    (df_amr_std if not df_amr_std.empty else pd.DataFrame(
        columns=["sequence","symbol","name","start","end","strand","pct_identity","pct_cov"]
    )).to_csv(amr_csv, index=False)

    # 3) Prodigal
    prodigal_done = False
    if not skip_prodigal:
        run_prodigal(fasta, sdir / sample, closed_circular=True)
        prodigal_done = True

    return {
        "sample": sample,
        "ori_csv": str(ori_csv),
        "amr_csv": str(amr_csv),
        "prodigal": prodigal_done,
        "n_ori_kept": int(0 if df_ori_final is None or df_ori_final.empty else len(df_ori_final)),
        "n_amr": int(0 if df_amr_std is None or df_amr_std.empty else len(df_amr_std)),
    }

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="QC pipeline for plasmid sequences.")
    ap.add_argument("--in", dest="in_path", required=True, help="FASTA file or directory of FASTAs")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--oridb_prefix", required=True, help="Prefix/path for ori BLAST DB (nucl)")
    ap.add_argument("--oridb_ref", default=None, help="FASTA of ori references to build DB if missing")
    ap.add_argument("--min_pident", type=float, default=85.0, help="Min % identity for ori hits")
    ap.add_argument("--min_scovs", type=float, default=80.0, help="Min subject % coverage for ori hits")
    ap.add_argument("--min_len", type=int, default=100, help="Min alignment length for ori hits (bp)")
    ap.add_argument("--threads", type=int, default=1, help="Threads for BLAST/AMRFinder")
    ap.add_argument("--skip_prodigal", action="store_true", help="Skip running Prodigal")
    args = ap.parse_args()

    in_path   = Path(args.in_path)
    outdir    = Path(args.outdir)
    db_prefix = Path(args.oridb_prefix)
    oridb_ref = Path(args.oridb_ref) if args.oridb_ref else None

    ensure_dir(outdir)
    # Ensure ori DB
    ensure_blast_db(oridb_ref, db_prefix)

    fastas = list_fastas(in_path)
    print(f"[INFO] Found {len(fastas)} FASTA(s).")

    # Per-seq processing
    summaries = []
    all_ori_rows = []
    all_amr_rows = []

    for fa in fastas:
        try:
            res = process_one(
                fasta=fa,
                outdir=outdir,
                db_prefix=db_prefix,
                min_pident=args.min_pident,
                min_scovs=args.min_scovs,
                min_len=args.min_len,
                threads=args.threads,
                skip_prodigal=args.skip_prodigal,
            )
            summaries.append(res)

            # Load and append per-seq results to aggregates
            ori_csv = Path(res["ori_csv"])
            amr_csv = Path(res["amr_csv"])
            if ori_csv.exists() and ori_csv.stat().st_size > 0:
                df = pd.read_csv(ori_csv)
                if not df.empty:
                    all_ori_rows.append(df)
            if amr_csv.exists() and amr_csv.stat().st_size > 0:
                df = pd.read_csv(amr_csv)
                if not df.empty:
                    all_amr_rows.append(df)
        except sp.CalledProcessError as e:
            print(f"[ERROR] Tool failed on {fa.name}: {e}", flush=True)
        except Exception as e:
            print(f"[ERROR] {fa.name}: {e}", flush=True)

    # Write aggregate CSVs
    agg_ori = pd.concat(all_ori_rows, ignore_index=True) if all_ori_rows else pd.DataFrame(
        columns=["sequence","ori_type","pct_identity","pct_cov_subject","q_start","q_end","strand","qlen","sstart","send","slen","length","bitscore","evalue"]
    )
    agg_amr = pd.concat(all_amr_rows, ignore_index=True) if all_amr_rows else pd.DataFrame(
        columns=["sequence","symbol","name","start","end","strand","pct_identity","pct_cov"]
    )
    agg_ori.to_csv(outdir / "aggregate_ori_calls.csv", index=False)
    agg_amr.to_csv(outdir / "aggregate_amr_calls.csv", index=False)

    pd.DataFrame(summaries).to_csv(outdir / "qc_summary.csv", index=False)
    print("[DONE] QC complete.")
    print(f"  - Aggregate ORIs: {outdir/'aggregate_ori_calls.csv'}")
    print(f"  - Aggregate AMRs: {outdir/'aggregate_amr_calls.csv'}")
    print(f"  - Summary:        {outdir/'qc_summary.csv'}")

if __name__ == "__main__":
    main()

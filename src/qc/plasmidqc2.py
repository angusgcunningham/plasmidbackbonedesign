#!/usr/bin/env python3
"""
PlasmidQC: oriV → ARGs → (extensible) pipeline

Steps:
  1) ORFs via Prodigal
  2) (Optional) MOB-typer (Inc/rep hits, oriT, mobility)
  3) oriV candidates:
       - anchor near replicon coords if available (MOB analysis)
       - else sliding windows with iteron-like repeats + AT-richness + rep proximity proxy
  4) (Optional) BLAST baseline vs a small oriV/replicon reference set
  5) ARGs via AMRFinderPlus (or ARGNet wrapper if present)
  6) Per-plasmid TSV + oriV candidates table in --workdir

External tools (PATH): prodigal, mob_typer, makeblastdb, blastn, amrfinder
"""
from __future__ import annotations
import os, sys, shutil, subprocess as sp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO

# -----------------------------
# UTILITIES
# -----------------------------
def which(tool: str) -> Optional[str]:
    return shutil.which(tool)

def run(cmd: List[str], cwd: Optional[str]=None, check: bool=True, capture: bool=False) -> sp.CompletedProcess:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    if capture:
        return sp.run(cmd, cwd=cwd, check=check, text=True, stdout=sp.PIPE, stderr=sp.PIPE)
    else:
        return sp.run(cmd, cwd=cwd, check=check)

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

# -----------------------------
# FASTA / BASIC STATS
# -----------------------------
def load_fasta_one(fasta_path: str) -> Tuple[str, str]:
    rec = next(SeqIO.parse(fasta_path, "fasta"))
    return str(rec.seq).upper(), rec.id

def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    at = seq.count('A') + seq.count('T')
    denom = gc + at
    return (gc/denom)*100 if denom else 0.0

# -----------------------------
# ORF CALLING (Prodigal)
# -----------------------------
def run_prodigal(plasmid_fasta: str, outdir: Path) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / Path(plasmid_fasta).stem
    faa = prefix.with_suffix(".proteins.faa")
    gff = prefix.with_suffix(".genes.gff")
    if not which("prodigal"):
        print("[WARN] Prodigal not found on PATH; skipping ORF calling.")
        return Path(), Path()
    cmd = ["prodigal", "-i", plasmid_fasta, "-a", str(faa), "-o", str(gff), "-p", "meta", "-q"]
    run(cmd)
    return faa, gff

# -----------------------------
# MOB-TYPER (replicon typing)
# -----------------------------
def run_mob_typer(plasmid_fasta: str, outdir: Path, relaxed: bool=False) -> Optional[pd.DataFrame]:
    if not which("mob_typer"):
        print("[WARN] mob_typer not found; skipping replicon typing.")
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / "mob_typer.tsv"
    analysis_dir = outdir / "analysis"

    cmd = [
        "mob_typer", "-i", plasmid_fasta, "-o", str(out_file),
        "-a", str(analysis_dir), "-n", "4"
    ]
    if relaxed:
        cmd += ["--min_rep_ident","60","--min_rep_cov","0.5","--min_rep_evalue","1e-5"]
    run(cmd)

    if out_file.exists():
        return pd.read_csv(out_file, sep="\t")
    tsvs = list(outdir.glob("*.tsv")) + list(analysis_dir.glob("*.tsv"))
    if tsvs:
        try:
            return pd.read_csv(tsvs[0], sep="\t")
        except Exception:
            pass
    print("[WARN] Could not locate MOB-typer TSV output; continuing without it.")
    return None

# -----------------------------
# ITERON / AT SCORING (oriV heuristics)
# -----------------------------
def hamming(s: str, t: str) -> int:
    return sum(a != b for a,b in zip(s, t))

def iteron_score(seq_window: str, k_range=range(17,23), max_mismatches=1, span: int=200) -> float:
    """Crude iteron-like repeat density in a local neighborhood. Returns ~0..1."""
    s = seq_window.upper()
    n = len(s)
    if n <= 0:
        return 0.0
    best = 0.0
    for k in k_range:
        counts = 0
        for i in range(0, max(0, n - k)):
            motif = s[i:i+k]
            lim = min(n - k, i + span)
            for j in range(i + k, lim):
                if hamming(motif, s[j:j+k]) <= max_mismatches:
                    counts += 1
        # normalize roughly by opportunities
        dens = counts / max(1, n / k)
        best = max(best, dens)
    return clamp01(best)

def at_rich_score(seq_window: str, subwin: int=200, stride: int=50) -> float:
    s = seq_window.upper()
    n = len(s)
    if n == 0:
        return 0.0
    best = 0.0
    for i in range(0, max(1, n - subwin + 1), stride):
        chunk = s[i:i+subwin]
        if not chunk:
            continue
        at = chunk.count('A') + chunk.count('T')
        best = max(best, at / len(chunk))
    return clamp01(best)

@dataclass
class OriVCandidate:
    start: int
    end: int
    score: float
    iteron: float
    atfrac: float
    prox: float
    meta: Dict

def score_window(seq: str, start: int, end: int, rep_cds_start: Optional[int]=None,
                 weights: Tuple[float,float,float]=(0.7, 0.2, 0.1)) -> Tuple[float, Dict[str,float]]:
    w_iter, w_at, w_prox = weights
    s = seq[start:end]
    iteron = clamp01(iteron_score(s))
    atp    = clamp01(at_rich_score(s))
    if rep_cds_start is None:
        prox = 0.5
    else:
        center = (start + end) // 2
        d = abs(center - rep_cds_start)
        prox = 1.0 / (1.0 + d/1000.0)
    prox = clamp01(prox)
    score = clamp01(w_iter*iteron + w_at*atp + w_prox*prox)
    parts = {"iteron": iteron, "at": atp, "prox": prox}
    return score, parts

# -----------------------------
# CANDIDATE WINDOWS
# -----------------------------
def windows_from_mob(df: Optional[pd.DataFrame], seq_len: int, flank_bp: int=2000) -> List[Tuple[int,int,Dict]]:
    windows = []
    if df is None or df.empty:
        return windows
    # Try obvious columns first (rarely present)
    start_cols = [c for c in df.columns if 'start' in c.lower() and ('rep' in c.lower() or 'repl' in c.lower())]
    end_cols   = [c for c in df.columns if 'end'   in c.lower() and ('rep' in c.lower() or 'repl' in c.lower())]
    type_cols  = [c for c in df.columns if 'replicon' in c.lower() and 'type' in c.lower()]
    for _, row in df.iterrows():
        s = None; e = None
        if start_cols and end_cols:
            try:
                s = int(row[start_cols[0]]); e = int(row[end_cols[0]])
            except Exception:
                s = None; e = None
        meta = {}
        if type_cols:
            meta['inc_groups'] = str(row[type_cols[0]])
        if s is not None and e is not None and 0 <= s < seq_len and 0 < e <= seq_len:
            start = max(0, min(s, e) - flank_bp)
            end   = min(seq_len, max(s, e) + flank_bp)
            meta["source"] = "mob_summary"
            windows.append((start, end, meta))
    return windows

def mob_replicon_windows_from_analysis(mob_outdir: Path, seq_len: int, flank_bp: int = 2000):
    """
    Use MOB-typer's replicon BLAST results to anchor oriV windows.
    File: <mob_outdir>/analysis/replicon_blast_results.txt
    Robust to column name variations.
    """
    analysis = mob_outdir / "analysis" / "replicon_blast_results.txt"
    if not analysis.exists():
        return []
    try:
        df = pd.read_csv(analysis, sep="\t")
    except Exception:
        return []
    lc = {c.lower(): c for c in df.columns}

    # Prefer subject coords (your plasmid)
    def pick_coords(row):
        for s_key in ["sstart","s_start","subject_start","subject.s","subject.sstart","subject.s.start"]:
            for e_key in ["send","s_end","subject_end","subject.e","subject.send","subject.e.end"]:
                if s_key in lc and e_key in lc:
                    try:
                        return int(row[lc[s_key]]), int(row[lc[e_key]])
                    except Exception:
                        pass
        for s_key in ["qstart","q_start","query_start"]:
            for e_key in ["qend","q_end","query_end"]:
                if s_key in lc and e_key in lc:
                    try:
                        return int(row[lc[s_key]]), int(row[lc[e_key]])
                    except Exception:
                        pass
        return None, None

    # Sort strongest first
    for col in ["bitscore","bit_score","score","pident","identity","length"]:
        if col in lc:
            df = df.sort_values(lc[col], ascending=False)
            break

    wins = []
    for _, r in df.head(5).iterrows():
        s, e = pick_coords(r)
        if s is None or e is None:
            continue
        start = max(0, min(s, e) - flank_bp)
        end   = min(seq_len, max(s, e) + flank_bp)
        meta = {"source": "mob_replicon"}
        for k in ["rep_type(s)","replicon_type","inc_group","cluster","rep_type"]:
            lk = k.lower()
            if lk in lc and pd.notna(r[lc[lk]]):
                meta["inc_groups"] = str(r[lc[lk]])
                break
        wins.append((start, end, meta))
    return wins

def fallback_windows(seq_len: int, wlen: int=3000, stride: int=500) -> List[Tuple[int,int,Dict]]:
    wins = []
    for i in range(0, seq_len, stride):
        start = i
        end = min(seq_len, i + wlen)
        wins.append((start, end, {"source": "fallback"}))
    return wins

# -----------------------------
# BLAST BASELINE AGAINST ORIV REFS
# -----------------------------
def ensure_blast_db(ref_fasta: Optional[str], db_prefix: Path) -> Optional[Path]:
    if ref_fasta is None:
        return None
    db_prefix = Path(db_prefix)
    if all((db_prefix.with_suffix(ext)).exists() for ext in [".nhr", ".nin", ".nsq"]):
        return db_prefix
    if not which("makeblastdb"):
        print("[WARN] makeblastdb not found; skipping BLAST baseline.")
        return None
    run(["makeblastdb", "-in", ref_fasta, "-dbtype", "nucl", "-out", str(db_prefix)])
    return db_prefix

@dataclass
class BlastHit:
    sseqid: str
    pident: float
    length: int
    evalue: float
    bitscore: float
    sstart: int
    send: int
    slen: int
    ref_cov: float

def blast_against_oriv(plasmid_fasta: str, db_prefix: Path, out_tsv: Path,
                       max_hits: int,
                       task: str,
                       min_ident: float, min_refcov: float, min_len: int, max_e: float) -> List[BlastHit]:
    if not which("blastn"):
        print("[WARN] blastn not found; skipping BLAST baseline.")
        return []
    outfmt = ("6 qseqid sseqid pident length evalue bitscore qstart qend sstart send qlen slen qcovs qcovhsp")
    cmd = ["blastn", "-task", task, "-query", plasmid_fasta,
           "-db", str(db_prefix), "-outfmt", outfmt, "-max_target_seqs", str(max_hits),
           "-evalue", str(max_e), "-out", str(out_tsv)]
    run(cmd)
    hits: List[BlastHit] = []
    if not out_tsv.exists():
        return hits
    cols = ["qseqid","sseqid","pident","length","evalue","bitscore",
            "qstart","qend","sstart","send","qlen","slen","qcovs","qcovhsp"]
    df = pd.read_csv(out_tsv, sep="\t", names=cols, header=None)
    df["ref_cov"] = df["length"] / df["slen"].replace(0, np.nan)
    keep = (
        (df["evalue"] <= max_e) &
        (df["pident"] >= min_ident) &
        ((df["ref_cov"] >= min_refcov) | (df["length"] >= min_len))
    )
    df = df[keep].sort_values(["bitscore","pident","length"], ascending=False)
    for _, r in df.head(max_hits).iterrows():
        hits.append(BlastHit(
            sseqid=str(r["sseqid"]),
            pident=float(r["pident"]),
            length=int(r["length"]),
            evalue=float(r["evalue"]),
            bitscore=float(r["bitscore"]),
            sstart=int(r["sstart"]),
            send=int(r["send"]),
            slen=int(r["slen"]),
            ref_cov=float(r["ref_cov"]) if pd.notna(r["ref_cov"]) else 0.0
        ))
    return hits

# -----------------------------
# ARG DETECTION
# -----------------------------
def run_amrfinder(proteins_faa: Path, out_tsv: Path) -> Optional[pd.DataFrame]:
    tool = which("amrfinder")  # amrfinderplus installs 'amrfinder'
    if not tool or not proteins_faa or not proteins_faa.exists():
        print("[WARN] AMRFinderPlus not available or proteins missing; skipping ARG detection.")
        return None
    for cmd in ([tool, "-p", str(proteins_faa), "-o", str(out_tsv)],
                [tool, "-p", str(proteins_faa), "--output", str(out_tsv)]):
        try:
            run(cmd)
            break
        except Exception as e:
            last_err = e
    if not out_tsv.exists():
        print(f"[WARN] AMRFinder failed; skipping. ({last_err})")
        return None
    try:
        return pd.read_csv(out_tsv, sep="\t", comment="#")
    except Exception:
        print("[WARN] Could not parse AMRFinder output.")
        return None

def extract_arg_symbols(amr_df: Optional[pd.DataFrame]) -> List[str]:
    if amr_df is None or amr_df.empty:
        return []
    # Try canonical "Element symbol", then fall back
    candidates = [c for c in amr_df.columns if c.lower().strip() in
                  {"element symbol","element_symbol","gene symbol","gene_symbol","symbol"} or
                  "element symbol" in c.lower()]
    if not candidates:
        return []
    col = candidates[0]
    vals = [str(x).strip() for x in amr_df[col].dropna().astype(str).tolist() if str(x).strip()]
    # de-duplicate preserving order
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

# (Optional) ARGNet wrapper
def run_argnet_on_proteins(proteins_faa: Path, out_dir: Path) -> Optional[pd.DataFrame]:
    if not Path("argnet.py").exists():
        print("[INFO] ARGNet script not found in CWD; skipping ARGNet.")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / (proteins_faa.stem + ".argnet.txt")
    cmd = [sys.executable, "argnet.py", "--input", str(proteins_faa),
           "--type", "aa", "--model", "argnet-l", "--outname", str(out_txt.name)]
    run(cmd, cwd=out_dir)
    try:
        return pd.read_csv(out_dir / out_txt.name, sep="\t")
    except Exception:
        return None

# -----------------------------
# MAIN: ORIV CANDIDATES & SUMMARY
# -----------------------------
def call_oriv(seq: str,
              mob_df: Optional[pd.DataFrame],
              blast_hits: List[BlastHit],
              mob_outdir: Optional[Path] = None) -> List[OriVCandidate]:
    seq_len = len(seq)
    wins = windows_from_mob(mob_df, seq_len, flank_bp=2000)

    # Mine MOB analysis if summary gave no coords
    if (not wins) and (mob_outdir is not None):
        extra = mob_replicon_windows_from_analysis(mob_outdir, seq_len, flank_bp=2000)
        if extra:
            wins = extra

    if not wins:
        print("[INFO] No replicon coords; using fallback sliding windows across contig.")
        wins = fallback_windows(seq_len, wlen=3000, stride=500)

    cands = []
    for (start, end, meta) in wins:
        score, parts = score_window(seq, start, end, rep_cds_start=None)
        cands.append(OriVCandidate(start, end, score, parts['iteron'], parts['at'], parts['prox'], meta))
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands

def extract_orit_type(mob_df: Optional[pd.DataFrame]) -> str:
    if mob_df is None or mob_df.empty:
        return ""
    for k in ["orit_type(s)", "orit_type", "orit"]:
        if k in mob_df.columns:
            vals = (mob_df[k].dropna().astype(str).tolist())
            if vals:
                return ",".join(sorted(set(",".join(vals).split(","))))
    return ""

def extract_inc_groups(mob_df: Optional[pd.DataFrame]) -> str:
    if mob_df is None or mob_df.empty:
        return ""
    for k in ["rep_type(s)","replicon_type","inc_group","rep_type"]:
        if k in mob_df.columns:
            vals = (mob_df[k].dropna().astype(str).tolist())
            if vals:
                return ",".join(sorted(set(",".join(vals).split(","))))
    return ""

def summarize_output(plasmid_id: str, seq: str,
                     mob_df: Optional[pd.DataFrame], oriv_cands: List[OriVCandidate],
                     blast_hits: List[BlastHit], amr_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    length_bp = len(seq)
    gc = gc_content(seq)
    inc = extract_inc_groups(mob_df)
    orit = extract_orit_type(mob_df)
    top = oriv_cands[0] if oriv_cands else None
    blast = blast_hits[0] if blast_hits else None
    arg_syms = extract_arg_symbols(amr_df)
    rows = [{
        "plasmid_id": plasmid_id,
        "length_bp": length_bp,
        "gc_percent": round(gc,2),
        "inc_groups": inc,
        "orit_type": orit,
        "oriv_start": top.start if top else "",
        "oriv_end": top.end if top else "",
        "oriv_score": round(top.score,4) if top else "",
        "oriv_iteron": round(top.iteron,4) if top else "",
        "oriv_ATmax": round(top.atfrac,4) if top else "",
        "oriv_prox": round(top.prox,4) if top else "",
        "blast_hit": (blast.sseqid if blast else ""),
        "blast_pident": (round(blast.pident,2) if blast else ""),
        "blast_ref_cov": (round(blast.ref_cov,3) if blast else ""),
        "blast_evalue": (blast.evalue if blast else ""),
        "blast_start": (blast.sstart if blast else ""),
        "blast_end": (blast.send if blast else ""),
        "arg_count": len(arg_syms),
        "arg_list": ";".join(arg_syms)
    }]
    return pd.DataFrame(rows)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="PlasmidQC: oriV → ARGs pipeline")
    ap.add_argument("--fasta", required=True, help="Input plasmid FASTA (single record)")
    ap.add_argument("--workdir", required=True, help="Output working directory")

    # oriV BLAST: either --oriv_db_prefix (prebuilt) OR --oriv_ref (FASTA to build once)
    ap.add_argument("--oriv_db_prefix", default=None,
                    help="Prefix of a prebuilt BLAST DB (e.g. /path/oriV_db). Skips makeblastdb.")
    ap.add_argument("--oriv_ref", default=None,
                    help="FASTA of oriV/replicon refs (if no prebuilt DB).")
    ap.add_argument("--skip_blast", action="store_true", help="Skip BLAST baseline")

    # BLAST thresholds
    ap.add_argument("--blast_task", default="blastn", choices=["blastn","dc-megablast"],
                    help="BLAST task (dc-megablast helps with divergent matches)")
    ap.add_argument("--blast_min_ident", type=float, default=80.0, help="Minimum identity (%)")
    ap.add_argument("--blast_min_refcov", type=float, default=0.60, help="Minimum reference coverage (0..1)")
    ap.add_argument("--blast_min_len", type=int, default=150, help="Minimum HSP length (bp)")
    ap.add_argument("--blast_evalue", type=float, default=1e-10, help="Max E-value")
    ap.add_argument("--blast_max_hits", type=int, default=5, help="Max hits to keep")

    # MOB / ARG
    ap.add_argument("--skip_mob", action="store_true", help="Skip MOB-typer")
    ap.add_argument("--mob_relaxed", action="store_true", help="Relax MOB-typer replicon thresholds")
    ap.add_argument("--arg_tool", choices=["amrfinder","argnet","none"], default="amrfinder",
                    help="ARG detector to run on Prodigal proteins")

    args = ap.parse_args()

    fasta = Path(args.fasta)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    seq, pid = load_fasta_one(str(fasta))

    # 1) ORFs
    proteins_faa, gff = run_prodigal(str(fasta), work)

    # 2) MOB-typer
    mob_dir = work / "mob"
    mob_df = None
    if not args.skip_mob:
        mob_df = run_mob_typer(str(fasta), mob_dir, relaxed=args.mob_relaxed)

    # 3) BLAST baseline (hits initialized)
    hits: List[BlastHit] = []
    db = None
    if not args.skip_blast:
        if args.oriv_db_prefix:
            pref = Path(args.oriv_db_prefix)
            needed = [pref.with_suffix(".nhr"), pref.with_suffix(".nin"), pref.with_suffix(".nsq")]
            if not all(p.exists() for p in needed):
                raise FileNotFoundError(f"Prebuilt DB missing components: {needed}")
            db = pref
        elif args.oriv_ref:
            local = work / "oriV_db"
            db = ensure_blast_db(args.oriv_ref, local)
        if db is not None:
            hits = blast_against_oriv(
                plasmid_fasta=str(fasta), db_prefix=db, out_tsv=work / "blast_oriv.tsv",
                max_hits=args.blast_max_hits, task=args.blast_task,
                min_ident=args.blast_min_ident, min_refcov=args.blast_min_refcov,
                min_len=args.blast_min_len, max_e=args.blast_evalue
            )

    # 4) oriV candidates (prefer MOB coords; else fallback)
    cands = call_oriv(seq, mob_df, hits, mob_outdir=(mob_dir if not args.skip_mob else None))

    # 5) ARGs
    amr_df = None
    if proteins_faa and proteins_faa.exists():
        if args.arg_tool == "amrfinder":
            amr_df = run_amrfinder(proteins_faa, work / "amrfinder.tsv")
        elif args.arg_tool == "argnet":
            amr_df = run_argnet_on_proteins(proteins_faa, work / "argnet")

    # 6) Summary (overwrite, not append)
    summary = summarize_output(pid, seq, mob_df, cands, hits, amr_df)
    out_tsv = work / "plasmid_qc_summary.tsv"
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Summary written: {out_tsv}")

    # oriV candidates table
    cand_rows = []
    for c in cands:
        row = {
            "start": c.start, "end": c.end, "score": round(c.score,4),
            "iteron": round(c.iteron,4), "ATmax": round(c.atfrac,4), "prox": round(c.prox,4),
            **{f"meta_{k}": v for k, v in c.meta.items()}
        }
        cand_rows.append(row)
    cand_df = pd.DataFrame(cand_rows)
    cand_path = work / "oriv_candidates.tsv"
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f".[OK] oriV candidates: {cand_path}")

if __name__ == "__main__":
    main()

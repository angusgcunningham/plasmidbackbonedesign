#!/usr/bin/env python3
# summarize_plannotate_gbk.py
# Usage:
#   python /cs/student/projects1/aibh/2024/acunning/Projects/Analysis/summarize_plannotate_gbk.py \
#     "/cs/student/projects1/aibh/2024/acunning/Projects/Results/plannotate/plan_basegfp2" \
#     --outdir "/cs/student/projects1/aibh/2024/acunning/Projects/Results/plannotate/summaries" \
#     --ext .gb .gbk \
#     --min_identity 95 --min_coverage 80 --exclude_fragments

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature

ABX_PATTERNS = [
    "ampr", "bla", "beta-lactamase",
    "kanr", "neor", "aph", "npt",
    "cmr", "cat",
    "specr", "smr", "aada",
    "tetr", "tetA", "tetR",
    "hygro", "hygr",
    "zeo", "zeor",
    "gmr", "gent", "aac",
]
REPORTER_PATTERNS = [
    "gfp", "egfp", "sfgfp", "bfp", "ebfp", "yfp", "rfp", "mcherry",
    "mneongreen", "venus", "cfp", "tdtomato"
]

def _q(qual: Dict[str, List[str]], key: str, default: str = "") -> str:
    vals = qual.get(key, [])
    return vals[0] if vals else default

def _to_float(x: str | float | None) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) or isinstance(x, int):
        return float(x)
    s = str(x).strip().replace("%", "")
    if s in {"", ".", "nan", "NaN", "None"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _strand_to_symbol(strand: Optional[int]) -> str:
    return "+" if strand == 1 else "-" if strand == -1 else "."

def parse_gbk(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    try:
        rec = SeqIO.read(str(path), "genbank")
    except Exception as e:
        print(f"[WARN] Failed to parse {path.name}: {e}", file=sys.stderr)
        return None, None

    plasmid_id = path.stem
    length_bp = len(rec.seq) if rec.seq else None
    topology = rec.annotations.get("topology") or ("circular" if "circular" in (rec.description or "").lower() else "")

    rows = []
    for feat in rec.features:
        if not isinstance(feat, SeqFeature):
            continue
        ftype = feat.type or ""
        start = int(feat.location.start)
        end = int(feat.location.end)
        length = int(len(feat.location))
        strand = _strand_to_symbol(feat.location.strand)

        q = feat.qualifiers or {}
        label = _q(q, "label")
        note = _q(q, "note")
        database = _q(q, "database")
        other = _q(q, "other")
        fragment = _q(q, "fragment").lower() == "true"
        identity = _to_float(_q(q, "identity"))
        match_len = _to_float(_q(q, "match_length"))  # pLannotate expresses this as percent

        rows.append({
            "plasmid_id": plasmid_id,
            "length_bp": length_bp,
            "topology": topology,
            "feature_type": ftype,
            "label": label,
            "note": note,
            "database": database,
            "other": other,
            "fragment": fragment,
            "identity": identity,
            "match_length": match_len,
            "start": start,
            "end": end,
            "feat_len": length,
            "strand": strand,
            "file": path.name,
        })

    feats = pd.DataFrame(rows)

    # Per-plasmid summary (ALL features)
    summary: Dict[str, Any] = {
        "plasmid_id": plasmid_id,
        "file": path.name,
        "length_bp": length_bp,
        "topology": topology or "",
        "num_features": len(feats),
        "num_cds": int((feats.feature_type == "CDS").sum()),
        "num_promoter": int((feats.feature_type == "promoter").sum()),
        "num_terminator": int((feats.feature_type == "terminator").sum()),
        "num_rep_origin": int((feats.feature_type == "rep_origin").sum()),
        "num_misc": int((feats.feature_type == "misc_feature").sum()),
        "num_fragments": int(feats.fragment.sum()) if len(feats) else 0,
        "num_full": int((~feats.fragment).sum()) if len(feats) else 0,
    }

    nonfrag = feats[~feats["fragment"]] if len(feats) else feats
    if len(nonfrag):
        summary["mean_identity_nonfrag"] = float(nonfrag.identity.dropna().mean()) if nonfrag.identity.notna().any() else None
        summary["min_identity_nonfrag"] = float(nonfrag.identity.dropna().min()) if nonfrag.identity.notna().any() else None
        summary["max_identity_nonfrag"] = float(nonfrag.identity.dropna().max()) if nonfrag.identity.notna().any() else None
    else:
        summary["mean_identity_nonfrag"] = summary["min_identity_nonfrag"] = summary["max_identity_nonfrag"] = None

    # ORIs list (ALL features)
    oris = feats.loc[feats.feature_type == "rep_origin", ["label"]]
    summary["ori_count"] = int(len(oris))
    summary["ori_labels"] = ";".join(sorted(set(oris["label"].dropna().astype(str)))) if len(oris) else ""

    # ABX markers (prefer non-fragment)
    abx_hits = feats[(feats.feature_type == "CDS")].assign(_lab=lambda d: d["label"].fillna("").str.lower())
    abx = set()
    for pat in ABX_PATTERNS:
        mask = abx_hits["_lab"].str.contains(pat, na=False)
        for _, r in abx_hits[mask].iterrows():
            abx.add(r["label"] or pat)
    if abx:
        nonfrag_abx = abx_hits[~abx_hits["fragment"]]
        if len(nonfrag_abx):
            abx = set(nonfrag_abx["label"].dropna().tolist()) or abx
    summary["abx_markers"] = ";".join(sorted(abx))

    # Reporters
    rep_hits = feats[(feats.feature_type == "CDS")].assign(_lab=lambda d: d["label"].fillna("").str.lower())
    reporters = set()
    for pat in REPORTER_PATTERNS:
        mask = rep_hits["_lab"].str.contains(pat, na=False)
        for _, r in rep_hits[mask].iterrows():
            reporters.add(r["label"] or pat)
    summary["reporters"] = ";".join(sorted(reporters))

    return feats, summary

def best_per_label(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best hit per (plasmid_id, label, feature_type): prefer non-fragment, then max identity, then longest match."""
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_frag_rank"] = tmp["fragment"].astype(int)  # 0 preferred
    tmp["_id_rank"] = -(tmp["identity"].fillna(-1e9))
    tmp["_len_rank"] = -(tmp["match_length"].fillna(-1e9))
    keys = ["plasmid_id", "label", "feature_type"]
    best = tmp.sort_values(keys + ["_frag_rank", "_id_rank", "_len_rank"]).groupby(keys, as_index=False).head(1)
    return best.drop(columns=["_frag_rank", "_id_rank", "_len_rank"])

def summarize_per_plasmid(feats: pd.DataFrame) -> pd.DataFrame:
    """Summarize any features table (can be filtered) into per-plasmid counts/labels."""
    if feats.empty:
        return pd.DataFrame(columns=[
            "plasmid_id","file","length_bp","topology","num_features","num_cds",
            "num_promoter","num_terminator","num_rep_origin","num_misc",
            "num_fragments","num_full","mean_identity_nonfrag","min_identity_nonfrag",
            "max_identity_nonfrag","ori_count","ori_labels","abx_markers","reporters"
        ])
    # Pull constant columns per plasmid if present
    base_cols = ["plasmid_id","file","length_bp","topology"]
    heads = (feats.sort_values(base_cols)
                  .groupby("plasmid_id", as_index=False).first()[base_cols])

    agg = feats.groupby("plasmid_id").agg(
        num_features=("feature_type","size"),
        num_cds=("feature_type", lambda s: (s=="CDS").sum()),
        num_promoter=("feature_type", lambda s: (s=="promoter").sum()),
        num_terminator=("feature_type", lambda s: (s=="terminator").sum()),
        num_rep_origin=("feature_type", lambda s: (s=="rep_origin").sum()),
        num_misc=("feature_type", lambda s: (s=="misc_feature").sum()),
        num_fragments=("fragment","sum")
    ).reset_index()
    agg["num_full"] = agg["num_features"] - agg["num_fragments"]

    # identities (non-frag only in this feats set)
    nonfrag = feats[~feats["fragment"]] if "fragment" in feats else feats
    def _safe_stat(x, fn, default=None):
        x = x.dropna()
        return fn(x) if len(x) else default
    ident = nonfrag.groupby("plasmid_id")["identity"].apply(lambda s: pd.Series({
        "mean_identity_nonfrag": _safe_stat(s, pd.Series.mean),
        "min_identity_nonfrag":  _safe_stat(s, pd.Series.min),
        "max_identity_nonfrag":  _safe_stat(s, pd.Series.max),
    })).reset_index()

    # ORI labels
    oris = feats[feats["feature_type"]=="rep_origin"].groupby("plasmid_id")["label"].apply(
        lambda s: ";".join(sorted(set(s.dropna().astype(str))))
    ).rename("ori_labels").reset_index()
    oric = feats[feats["feature_type"]=="rep_origin"].groupby("plasmid_id").size().rename("ori_count").reset_index()

    # ABX markers
    abx_hits = feats[feats["feature_type"]=="CDS"].assign(_lab=lambda d: d["label"].fillna("").str.lower())
    abx = (abx_hits.groupby("plasmid_id").apply(
        lambda d: ";".join(sorted(set(
            d.loc[sum([d["_lab"].str.contains(pat, na=False) for pat in ABX_PATTERNS]).astype(bool), "label"]
             .dropna().astype(str)
        )))
    ).rename("abx_markers").reset_index())

    # Reporters
    reps = (abx_hits.groupby("plasmid_id").apply(
        lambda d: ";".join(sorted(set(
            d.loc[sum([d["_lab"].str.contains(pat, na=False) for pat in REPORTER_PATTERNS]).astype(bool), "label"]
             .dropna().astype(str)
        )))
    ).rename("reporters").reset_index())

    out = heads.merge(agg, on="plasmid_id", how="left") \
               .merge(ident, on="plasmid_id", how="left") \
               .merge(oric, on="plasmid_id", how="left") \
               .merge(oris, on="plasmid_id", how="left") \
               .merge(abx, on="plasmid_id", how="left") \
               .merge(reps, on="plasmid_id", how="left")

    for c in ["ori_count","num_fragments","num_full","num_features","num_cds","num_promoter","num_terminator","num_rep_origin","num_misc"]:
        if c in out:
            out[c] = out[c].fillna(0).astype(int)
    return out

def filter_highconf(df: pd.DataFrame, min_identity: float, min_coverage: float, exclude_fragments: bool) -> pd.DataFrame:
    if df.empty:
        return df
    conds = pd.Series(True, index=df.index)
    if exclude_fragments and "fragment" in df.columns:
        conds &= ~df["fragment"].astype(bool)
    if "identity" in df.columns and min_identity is not None:
        conds &= df["identity"].fillna(0) >= float(min_identity)
    if "match_length" in df.columns and min_coverage is not None:
        conds &= df["match_length"].fillna(0) >= float(min_coverage)
    return df.loc[conds].copy()

def main():
    ap = argparse.ArgumentParser(description="Summarize pLannotate GenBank outputs into CSVs (full + high-confidence filtered).")
    ap.add_argument("indir", type=str, help="Directory containing .gb/.gbk outputs")
    ap.add_argument("--outdir", type=str, default=None, help="Where to write CSVs (default: <indir>/summary)")
    ap.add_argument("--ext", nargs="*", default=[".gb", ".gbk"], help="File extensions to include")
    ap.add_argument("--min_identity", type=float, default=95.0, help="Min identity (%) for high-confidence")
    ap.add_argument("--min_coverage", type=float, default=80.0, help="Min match_length/coverage (%) for high-confidence")
    ap.add_argument("--exclude_fragments", action="store_true", default=True, help="Exclude features with /fragment=True in high-confidence set")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    gbks = [p for p in indir.iterdir() if p.suffix.lower() in set(x.lower() for x in args.ext)]
    if not gbks:
        print(f"[ERR] No GenBank files with extensions {args.ext} found in {indir}", file=sys.stderr)
        sys.exit(1)

    all_feats = []
    summaries = []
    for p in sorted(gbks):
        feats, summ = parse_gbk(p)
        if feats is None:
            continue
        all_feats.append(feats)
        summaries.append(summ)

    feats_df = pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()
    summ_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()

    # Best-per-label (all)
    best_all = best_per_label(feats_df) if not feats_df.empty else pd.DataFrame()

    # High-confidence filtered tables
    feats_hi = filter_highconf(feats_df, args.min_identity, args.min_coverage, args.exclude_fragments)
    best_hi  = best_per_label(feats_hi) if not feats_hi.empty else pd.DataFrame()
    summ_hi  = summarize_per_plasmid(feats_hi)

    # Write CSVs
    (outdir / "features_all.csv").write_text(feats_df.to_csv(index=False))
    (outdir / "features_best_per_label.csv").write_text(best_all.to_csv(index=False))
    (outdir / "plasmid_summary.csv").write_text(summ_df.to_csv(index=False))

    (outdir / "features_highconf.csv").write_text(feats_hi.to_csv(index=False))
    (outdir / "features_best_per_label_highconf.csv").write_text(best_hi.to_csv(index=False))
    (outdir / "plasmid_summary_highconf.csv").write_text(summ_hi.to_csv(index=False))

    print("[OK] Wrote:")
    print(f"  {outdir / 'features_all.csv'}")
    print(f"  {outdir / 'features_best_per_label.csv'}")
    print(f"  {outdir / 'plasmid_summary.csv'}")
    print(f"  {outdir / 'features_highconf.csv'}")
    print(f"  {outdir / 'features_best_per_label_highconf.csv'}")
    print(f"  {outdir / 'plasmid_summary_highconf.csv'}")

if __name__ == "__main__":
    main()

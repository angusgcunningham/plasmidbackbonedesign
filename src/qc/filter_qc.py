#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def _fmt_list(vals, digits):
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("")
        elif isinstance(v, (int, np.integer)):
            out.append(str(v))
        elif isinstance(v, (float, np.floating)):
            out.append(f"{v:.{digits}f}")
        else:
            out.append(str(v))
    return ",".join(out)

def _load_oris(ori_csv: Path) -> pd.DataFrame:
    if not ori_csv.exists() or ori_csv.stat().st_size == 0:
        return pd.DataFrame(columns=["sequence","ori_type","pct_identity","pct_cov_subject","q_start","q_end"])
    df = pd.read_csv(ori_csv)
    # normalize column names
    if "ori_type" not in df.columns and "sseqid" in df.columns:
        df = df.rename(columns={"sseqid": "ori_type"})
    if "pct_identity" not in df.columns and "pident" in df.columns:
        df = df.rename(columns={"pident": "pct_identity"})
    # subject coverage may appear as pct_cov_subject or scovs
    if "pct_cov_subject" not in df.columns and "scovs" in df.columns:
        df = df.rename(columns={"scovs": "pct_cov_subject"})
    # query coords optional; for stable sorting if present
    for c in ("pct_identity","pct_cov_subject","q_start","q_end"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_amrs(amr_csv: Path) -> pd.DataFrame:
    if not amr_csv.exists() or amr_csv.stat().st_size == 0:
        return pd.DataFrame(columns=["sequence","symbol","name","pct_identity","pct_cov"])
    df = pd.read_csv(amr_csv)
    for c in ("pct_identity","pct_cov"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def filter_qc(
    qc_out: Path,
    out_pass_csv: Path,
    out_fail_csv: Path,
    ori_min_id: float | None,
    ori_min_cov: float | None,
    amr_min_id: float | None,
    amr_min_cov: float | None,
    ori_count_min: int | None,
    ori_count_max: int | None,
    amr_count_min: int | None,
    amr_count_max: int | None,
    digits: int = 2,
):
    # default to aggregate files in qc_out unless explicit paths were provided as full filenames
    ori_csv = qc_out if qc_out.suffix == ".csv" else qc_out / "aggregate_ori_calls.csv"
    amr_csv = qc_out if qc_out.suffix == ".csv" else qc_out / "aggregate_amr_calls.csv"

    if qc_out.suffix == ".csv":
        # If a single CSV was passed, assume it's ORI; try to find AMR alongside
        amr_csv = Path(str(qc_out).replace("ori", "amr"))

    odf = _load_oris(Path(ori_csv))
    adf = _load_amrs(Path(amr_csv))

    # union of plasmids seen in either table
    plasmids = sorted(set(odf.get("sequence", pd.Series([], dtype=str))).union(
                      set(adf.get("sequence", pd.Series([], dtype=str)))))

    passed_rows = []
    failed_rows = []

    for pid in plasmids:
        # ORIs for this plasmid
        o = odf.loc[odf["sequence"] == pid].copy() if not odf.empty else pd.DataFrame(columns=odf.columns)
        # stable order: by genomic start if present
        if {"q_start","q_end"}.issubset(o.columns):
            o = o.sort_values(["q_start","q_end"])
        # threshold filter (keep rows that meet thresholds, if thresholds given)
        if ori_min_id is not None:
            o = o[o["pct_identity"] >= ori_min_id]
        if ori_min_cov is not None and "pct_cov_subject" in o.columns:
            o = o[o["pct_cov_subject"] >= ori_min_cov]

        n_ori = len(o)

        # AMRs for this plasmid
        a = adf.loc[adf["sequence"] == pid].copy() if not adf.empty else pd.DataFrame(columns=adf.columns)
        # label preference: symbol → name
        label_series = a["symbol"] if "symbol" in a.columns else pd.Series([], dtype=str)
        if label_series.empty or (label_series.fillna("") == "").all():
            label_series = a["name"] if "name" in a.columns else pd.Series([], dtype=str)
        # threshold filter
        if amr_min_id is not None and "pct_identity" in a.columns:
            a = a[a["pct_identity"] >= amr_min_id]
        if amr_min_cov is not None and "pct_cov" in a.columns:
            a = a[a["pct_cov"] >= amr_min_cov]

        n_amr = len(a)

        # evaluate pass/fail
        reasons = []

        # ORI presence/count
        if ori_count_min is not None and n_ori < ori_count_min:
            reasons.append(f"ORI count {n_ori} < min {ori_count_min}")
        if ori_count_max is not None and n_ori > ori_count_max:
            reasons.append(f"ORI count {n_ori} > max {ori_count_max}")
        if ori_count_min is None and ori_count_max is None and n_ori == 0:
            reasons.append("No ORI")

        # AMR presence/count
        if amr_count_min is not None and n_amr < amr_count_min:
            reasons.append(f"ARG count {n_amr} < min {amr_count_min}")
        if amr_count_max is not None and n_amr > amr_count_max:
            reasons.append(f"ARG count {n_amr} > max {amr_count_max}")
        if amr_count_min is None and amr_count_max is None and n_amr == 0:
            reasons.append("No ARG")

        if reasons:
            failed_rows.append({"Plasmid_ID": pid, "reason failed": "; ".join(reasons)})
            continue

        # build passed row
        ori_names = o["ori_type"].fillna("").astype(str).tolist() if "ori_type" in o.columns else []
        ori_ids   = o["pct_identity"].tolist() if "pct_identity" in o.columns else []
        ori_covs  = o["pct_cov_subject"].tolist() if "pct_cov_subject" in o.columns else []

        # refresh AMR labels after filtering
        labels = (a["symbol"] if "symbol" in a.columns else a.get("name", pd.Series([], dtype=str))).fillna("").astype(str).tolist()
        amr_ids = a["pct_identity"].tolist() if "pct_identity" in a.columns else []
        amr_cov = a["pct_cov"].tolist() if "pct_cov" in a.columns else []

        passed_rows.append({
            "Plasmid_ID": pid,
            "Ori's present": _fmt_list(ori_names, digits),
            "Identity of each ori": _fmt_list(ori_ids, digits),
            "Cov of each ori": _fmt_list(ori_covs, digits),
            "ARG's present": _fmt_list(labels, digits),
            "Identity of ARGs": _fmt_list(amr_ids, digits),
            "Cov of ARGs": _fmt_list(amr_cov, digits),
        })

    pd.DataFrame(passed_rows).sort_values("Plasmid_ID").to_csv(out_pass_csv, index=False)
    pd.DataFrame(failed_rows).sort_values("Plasmid_ID").to_csv(out_fail_csv, index=False)

def parse_args():
    ap = argparse.ArgumentParser(description="Filter plasmids based on ORI/ARG thresholds and counts.")
    ap.add_argument("--qc_out", required=True,
                    help="QC output directory containing aggregate_ori_calls.csv and aggregate_amr_calls.csv")
    ap.add_argument("--out_pass", required=True, help="CSV path for passed plasmids")
    ap.add_argument("--out_fail", required=True, help="CSV path for failed plasmids")

    # thresholds (None = no threshold)
    ap.add_argument("--ori_min_identity", type=float, default=None, help="Min %% identity for ORI hits")
    ap.add_argument("--ori_min_cov",      type=float, default=None, help="Min %% subject coverage for ORI hits")
    ap.add_argument("--amr_min_identity", type=float, default=None, help="Min %% identity for ARG hits")
    ap.add_argument("--amr_min_cov",      type=float, default=None, help="Min %% coverage for ARG hits")

    # count windows (use min=max for exact; e.g., 1..1 means exactly 1; 1..2 means 1 or 2)
    ap.add_argument("--ori_count_min", type=int, default=None, help="Minimum number of ORIs required (after thresholds)")
    ap.add_argument("--ori_count_max", type=int, default=None, help="Maximum number of ORIs allowed (after thresholds)")
    ap.add_argument("--amr_count_min", type=int, default=None, help="Minimum number of ARGs required (after thresholds)")
    ap.add_argument("--amr_count_max", type=int, default=None, help="Maximum number of ARGs allowed (after thresholds)")

    ap.add_argument("--digits", type=int, default=2, help="Decimal places for % fields in outputs")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    filter_qc(
        qc_out=Path(args.qc_out),
        out_pass_csv=Path(args.out_pass),
        out_fail_csv=Path(args.out_fail),
        ori_min_id=args.ori_min_identity,
        ori_min_cov=args.ori_min_cov,
        amr_min_id=args.amr_min_identity,
        amr_min_cov=args.amr_min_cov,
        ori_count_min=args.ori_count_min,
        ori_count_max=args.ori_count_max,
        amr_count_min=args.amr_count_min,
        amr_count_max=args.amr_count_max,
        digits=args.digits,
    )

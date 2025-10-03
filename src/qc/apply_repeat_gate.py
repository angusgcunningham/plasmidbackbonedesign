#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

_EXT_RE = re.compile(r"\.(fa|fasta|fna|fas|gbk|gb|fa\.gz|fasta\.gz|fna\.gz|fas\.gz)$", re.IGNORECASE)

def _basename_no_ext(x: str) -> str:
    x = str(x).strip()
    x = Path(x).name
    if x.lower().endswith(".gz"):
        x = x[:-3]
    x = _EXT_RE.sub("", x)
    return x

def _norm_id(x: str) -> str:
    return _basename_no_ext(x).lower()

def _guess_id_col(df: pd.DataFrame, prefs=("Plasmid_ID","plasmid_id","sequence","id")) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for p in prefs:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    for c in df.columns:
        cl = c.lower()
        if "plas" in cl and "id" in cl:
            return c
    return df.columns[0] if len(df.columns) else "Plasmid_ID"

def enforce_repeat_gate(
    passed_csv: str,
    failed_csv: str,
    repeats_csv: str,
    repeat_max_len: int = 50,
    reason_text: str | None = None,
    use_ge: bool = False,  # set True if you want >= instead of >
):
    passed_path = Path(passed_csv)
    failed_path = Path(failed_csv)
    rpt_path    = Path(repeats_csv)

    if not passed_path.exists():
        raise FileNotFoundError(f"Passed CSV not found: {passed_path}")
    if not rpt_path.exists():
        raise FileNotFoundError(f"Repeats CSV not found: {rpt_path}")

    passed = pd.read_csv(passed_path)
    failed = (pd.read_csv(failed_path)
              if failed_path.exists() and failed_path.stat().st_size > 0
              else pd.DataFrame(columns=["Plasmid_ID","reason failed"]))

    rpt = pd.read_csv(rpt_path)

    # Detect ID columns
    pass_id_col = _guess_id_col(passed)
    fail_id_col = _guess_id_col(failed) if not failed.empty else "Plasmid_ID"
    rpt_id_col  = _guess_id_col(rpt, ("plasmid_id","Plasmid_ID","sequence","id"))

    # Normalize IDs
    passed["_idnorm"] = passed[pass_id_col].astype(str).map(_norm_id)
    if not failed.empty:
        failed["_idnorm"] = failed[fail_id_col].astype(str).map(_norm_id)

    # Repeats → bad IDs
    if "longest_len" not in rpt.columns:
        raise ValueError("Repeats CSV must contain 'longest_len' column.")
    rpt["longest_len"] = pd.to_numeric(rpt["longest_len"], errors="coerce").fillna(-1)

    rpt["_idnorm_from_col"]  = rpt[rpt_id_col].astype(str).map(_norm_id)
    rpt["_idnorm_from_file"] = rpt["file"].astype(str).map(_norm_id) if "file" in rpt.columns else ""

    if use_ge:
        bad_rows = rpt[rpt["longest_len"] >= repeat_max_len]
    else:
        bad_rows = rpt[rpt["longest_len"] >  repeat_max_len]

    bad_ids = set(bad_rows["_idnorm_from_col"].tolist()) | set(bad_rows["_idnorm_from_file"].tolist())
    bad_ids.discard("")

    reason = reason_text or f"repeat > {repeat_max_len}" if not use_ge else f"repeat >= {repeat_max_len}"

    # Intersections
    passed_ids = set(passed["_idnorm"].tolist())
    failed_ids = set(failed["_idnorm"].tolist()) if not failed.empty else set()

    bad_in_pass  = bad_ids & passed_ids
    bad_in_fail  = bad_ids & failed_ids
    bad_in_neither = bad_ids - passed_ids - failed_ids

    # A) move from PASSED → FAILED
    to_fail = passed[passed["_idnorm"].isin(bad_in_pass)].copy()
    keep    = passed[~passed["_idnorm"].isin(bad_in_pass)].copy()

    # Ensure failed has the helper col
    if failed.empty:
        failed = pd.DataFrame(columns=["Plasmid_ID","reason failed","_idnorm"])

    # add/append reasons for moved rows
    for _, r in to_fail.iterrows():
        pid_disp = r[pass_id_col]
        nid = r["_idnorm"]
        if (failed["_idnorm"] == nid).any():
            idx = failed["_idnorm"] == nid
            prev = failed.loc[idx, "reason failed"].fillna("").astype(str)
            failed.loc[idx, "reason failed"] = prev.apply(lambda s: s if reason in s else (s + "; " + reason if s else reason))
        else:
            failed = pd.concat([
                failed,
                pd.DataFrame([{"Plasmid_ID": pid_disp, "reason failed": reason, "_idnorm": nid}])
            ], ignore_index=True)

    # B) append reason to those already failed
    for nid in sorted(bad_in_fail):
        idx = failed["_idnorm"] == nid
        prev = failed.loc[idx, "reason failed"].fillna("").astype(str)
        failed.loc[idx, "reason failed"] = prev.apply(lambda s: s if reason in s else (s + "; " + reason if s else reason))

    # C) optionally log those neither in pass nor fail (often unprocessed/missing)
    if bad_in_neither:
        sample = list(sorted(bad_in_neither))[:5]
        print(f"[INFO] {len(bad_in_neither)} repeat-flagged IDs were in neither PASSED nor FAILED. Example (normalized): {sample}")

    # Write back
    keep.drop(columns=["_idnorm"], errors="ignore").to_csv(passed_path, index=False)
    failed.sort_values("Plasmid_ID").drop(columns=["_idnorm"], errors="ignore").to_csv(failed_path, index=False)

    print(f"[OK] Repeat gate applied. Moved {len(to_fail)} from PASSED → FAILED; "
          f"appended reason to {len(bad_in_fail)} already-failed rows. "
          f"Remaining passed: {len(keep)}; total failed rows: {len(failed)}.")

def main():
    ap = argparse.ArgumentParser(description="Move plasmids with long repeats from passed to failed; append reason for already-failed ones.")
    ap.add_argument("--passed_csv", required=True)
    ap.add_argument("--failed_csv", required=True)
    ap.add_argument("--repeats_csv", required=True)
    ap.add_argument("--repeat_max_len", type=int, default=50, help="Threshold in bp")
    ap.add_argument("--reason_text", default=None, help="Custom reason (default auto-generated)")
    ap.add_argument("--ge", action="store_true", help="Use '>=' instead of '>'")
    args = ap.parse_args()
    enforce_repeat_gate(
        args.passed_csv, args.failed_csv, args.repeats_csv,
        repeat_max_len=args.repeat_max_len, reason_text=args.reason_text, use_ge=args.ge
    )

if __name__ == "__main__":
    main()

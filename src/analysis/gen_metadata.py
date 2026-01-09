from pathlib import Path
from typing import Optional
from Bio import SeqIO
import pandas as pd

def build_generated_metadata(fasta_dir, out_file: Optional[Path] = None):
    """
    Parse all FASTA files in a directory, compute length and GC%, 
    and save metadata as <foldername>_generated_metadata.csv
    """
    fasta_dir = Path(fasta_dir)
    assert fasta_dir.exists(), f"Folder not found: {fasta_dir}"
    
    records = []
    for fasta_file in fasta_dir.glob("*.fasta"):
        for rec in SeqIO.parse(fasta_file, "fasta"):
            seq = str(rec.seq).upper()
            length = len(seq)
            gc = (seq.count("G") + seq.count("C")) / length if length > 0 else 0
            records.append({
                "plasmid_id": rec.id,
                "length_bp": length,
                "gc_percent": gc * 100,
                "source_file": fasta_file.name
            })
    
    df = pd.DataFrame(records)
    out_file = out_file or (fasta_dir.parent / f"{fasta_dir.name}_generated_metadata.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved metadata for {len(df)} sequences → {out_file}")
    return df

DEFAULT_FASTA_DIRS = [
    "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_base_atg",
    "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft35_atg",
]


def main() -> None:
    import argparse
    from datetime import datetime

    ap = argparse.ArgumentParser(description="Build metadata CSVs for generated FASTA dirs.")
    ap.add_argument("--fasta-dir", action="append", dest="fasta_dirs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    if args.fasta_dirs:
        fasta_dirs = args.fasta_dirs
    elif args.run_name:
        fasta_dirs = [str(Path("runs") / args.run_name / "generations")]
    else:
        fasta_dirs = DEFAULT_FASTA_DIRS

    out_csv = Path(args.out_csv) if args.out_csv else None
    if out_csv and len(fasta_dirs) == 1:
        build_generated_metadata(fasta_dirs[0], out_file=out_csv)
    else:
        for fasta_dir in fasta_dirs:
            build_generated_metadata(fasta_dir)


if __name__ == "__main__":
    main()

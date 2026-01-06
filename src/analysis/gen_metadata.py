from pathlib import Path
from Bio import SeqIO
import pandas as pd

def build_generated_metadata(fasta_dir):
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
    out_file = fasta_dir.parent / f"{fasta_dir.name}_generated_metadata.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved metadata for {len(df)} sequences → {out_file}")
    return df

# ─── Example Run ──────────────────────────────────────────────
meta = build_generated_metadata(
    "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_base_atg"
)
meta2 = build_generated_metadata(
    "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft35_atg"
)
import subprocess
import pandas as pd
from pathlib import Path

# Test on your strict QC passed sequences
DB_PATH = "/cs/student/projects1/aibh/2024/acunning/plasmid_db"

def calculate_true_similarity(query_fasta, db_path):
    """Calculate weighted similarity per position"""
    blast_cmd = [
        "blastn",
        "-query", str(query_fasta),
        "-db", db_path,
        "-outfmt", "6 qseqid sseqid pident length qlen slen qstart qend sstart send",
        "-max_target_seqs", "5",
        "-task", "megablast"
    ]
    
    result = subprocess.run(blast_cmd, 
                          capture_output=True, 
                          text=True, 
                          check=True)
    
    query_len = None
    # Track identity at each position
    position_identities = {}  # pos -> best_identity
    
    for line in result.stdout.splitlines():
        qid, sid, pident, length, qlen, slen, qstart, qend, sstart, send = line.split()
        
        if query_len is None:
            query_len = int(qlen)
            
        pident = float(pident)
        start, end = int(qstart), int(qend)
        
        # Store best identity for each position
        for pos in range(start, end + 1):
            if pos not in position_identities or pident > position_identities[pos]:
                position_identities[pos] = pident
    
    if not query_len or not position_identities:
        return 0.0
    
    if not query_len or not position_identities:
        return 0.0
    
    # Calculate and print detailed metrics
    total_identity = sum(position_identities.values())
    coverage = len(position_identities) / query_len
    avg_identity = total_identity / len(position_identities)
    weighted_similarity = (avg_identity * coverage) / 100
    
    print(f"\nDetailed metrics for {query_fasta.name}:")
    print(f"Total sequence length: {query_len}")
    print(f"Positions covered: {len(position_identities)}")
    print(f"Coverage: {coverage:.3f}")
    print(f"Average identity of covered positions: {avg_identity:.2f}%")
    print(f"Weighted similarity score: {weighted_similarity:.3f}")
    print("-" * 50)
    
    return weighted_similarity

from pathlib import Path
import pandas as pd
import subprocess

# --- Helper: compute per-position weighted similarity from BLAST tab lines ---
def _weighted_similarity_from_lines(lines):
    """
    lines: iterable of strings in outfmt 6:
      qseqid sseqid pident length qlen slen qstart qend sstart send [bitscore optional]
    Returns (weighted_similarity, coverage, avg_identity, qlen)
    """
    position_identities = {}
    query_len = None

    for line in lines:
        parts = line.split()
        # tolerate presence/absence of bitscore at the end
        qid, sid, pident, length, qlen, slen, qstart, qend = parts[:8]
        pident = float(pident)
        qstart, qend = int(qstart), int(qend)
        if query_len is None:
            query_len = int(qlen)
        # accumulate best identity per covered position
        # (BLAST coords are 1-based and inclusive)
        rng = range(qstart, qend + 1) if qstart <= qend else range(qend, qstart + 1)
        for pos in rng:
            if pos not in position_identities or pident > position_identities[pos]:
                position_identities[pos] = pident

    if not query_len or not position_identities:
        return 0.0, 0.0, 0.0, query_len or 0

    total_identity = sum(position_identities.values())
    coverage = len(position_identities) / query_len
    avg_identity = total_identity / len(position_identities)
    weighted_similarity = (avg_identity * coverage) / 100.0
    return weighted_similarity, coverage, avg_identity, query_len


def find_closest_match_weighted(query_fasta: Path, db_path: str):
    """
    Find the single best subject (by BLAST default ranking) and compute
    a per-position weighted similarity using *all HSPs for that subject*.
    Returns dict with subject_id, closest_weighted_similarity, closest_coverage, closest_avg_identity.
    """
    blast_cmd = [
        "blastn",
        "-query", str(query_fasta),
        "-db", db_path,
        "-outfmt", "6 qseqid sseqid pident length qlen slen qstart qend sstart send bitscore",
        "-max_target_seqs", "1",   # top subject only (may include multiple HSPs)
        "-task", "megablast",
        "-word_size", "28"
    ]
    res = subprocess.run(blast_cmd, capture_output=True, text=True, check=True)
    if not res.stdout.strip():
        return None

    lines = res.stdout.strip().splitlines()
    # All lines should correspond to the same top subject; take it from the first line
    first = lines[0].split()
    subject_id = first[1]

    weighted, cov, avg_id, qlen = _weighted_similarity_from_lines(lines)
    return {
        "subject_id": subject_id,
        "closest_weighted_similarity": weighted,
        "closest_coverage": cov,
        "closest_avg_identity": avg_id
    }

# === Batch over generated folders, writing one CSV per dataset ===
GEN_FOLDERS = {
    "base_atg": "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_base_atg",
    "base_gc":  "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_basegfp2",
    "ft15_atg": "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft15_atg",
    "ft15_gc":  "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft15kgfp_p",
    "ft35_atg": "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft35_atg",
    "ft35_gc":  "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft35kgfp3",
}

OUTDIR = Path("/cs/student/projects1/aibh/2024/acunning/Projects/Results/similarity_csvs")
OUTDIR.mkdir(parents=True, exist_ok=True)

for name, folder in GEN_FOLDERS.items():
    print(f"\nProcessing {name} …")
    records = []
    for fasta in Path(folder).glob("*.fa*"):  # matches .fa and .fasta
        plasmid_id = fasta.stem  # e.g., ft15k_generated_sequence_001

        # global (top-5) weighted similarity across all subjects/HSPs
        true_sim = calculate_true_similarity(fasta, DB_PATH)

        # closest subject weighted similarity (aggregate all HSPs for that subject)
        best = find_closest_match_weighted(fasta, DB_PATH)

        row = {
            "Plasmid_ID": plasmid_id,
            "true_similarity": true_sim,
        }
        if best:
            row.update({
                "closest_match": best["subject_id"],
                "closest_weighted_similarity": best["closest_weighted_similarity"],
                "closest_coverage": best["closest_coverage"],
                "closest_avg_identity": best["closest_avg_identity"],
            })
        records.append(row)

    df_out = pd.DataFrame(records)
    out_path = OUTDIR / f"{name}_similarity.csv"
    df_out.to_csv(out_path, index=False)
    print(f"→ Saved {len(df_out)} rows to {out_path}")

import os
import subprocess
from pathlib import Path
from Bio import SeqIO

DEFAULT_FT_DIR = "/cs/student/projects1/aibh/2024/acunning/Projects/Results/gen_ft15_atg"
DEFAULT_JSON_DIR = "/cs/student/projects1/aibh/2024/acunning/Projects/Results/plannotate/plan_ft15_atg"


def main() -> None:
    import argparse
    from datetime import datetime
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Batch pLannotate over FASTA directory.")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.run_name:
        run_dir = Path("runs") / args.run_name
        ft_dir = args.input_dir or str(run_dir / "generations")
        json_dir = args.output_dir or str(run_dir / "analysis" / "plannotate")
    else:
        ft_dir = args.input_dir or DEFAULT_FT_DIR
        json_dir = args.output_dir or DEFAULT_JSON_DIR
    os.makedirs(json_dir, exist_ok=True)

    # Get list of already processed files
    processed_files = {f.stem for f in Path(json_dir).glob("*.json")}
    failed_files = []

    for fname in os.listdir(ft_dir):
        if fname.endswith(".txt"):
            continue

        # Skip already processed files
        stem = Path(fname).stem
        if stem in processed_files:
            print(f"Skipping already processed file: {fname}")
            continue

        fasta_file = os.path.join(ft_dir, fname)
        try:
            print(f"Processing: {fname}")
            subprocess.run([
                "plannotate", "batch",
                "-i", fasta_file,
                "-o", json_dir
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to process {fname}")
            failed_files.append(fname)
            continue

    print("ft pLannotate finished →", json_dir)
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for f in failed_files:
            print(f"  - {f}")

'''
base_dir = "/cs/student/projects1/aibh/2024/acunning/Projects/Data/dataset/fasta"
json2_dir = "/cs/student/projects1/aibh/2024/acunning/Projects/Results/plannotate/training15k"
os.makedirs(json2_dir, exist_ok=True)

for fname in os.listdir(base_dir):
    if fname.endswith(".txt"):
        continue
    fasta_file = os.path.join(base_dir, fname)
    # here we use the correct subcommand: `plannotate batch -i file -o dir`
    subprocess.run([
        "plannotate", "batch",
        "-i", fasta_file,
        "-o", json2_dir
    ], check=True)

print("base pLannotate finished →", json2_dir)
'''


if __name__ == "__main__":
    main()

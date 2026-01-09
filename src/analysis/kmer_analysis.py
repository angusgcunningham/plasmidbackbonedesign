# kmer_analysis.py

from collections import Counter
import math
import matplotlib.pyplot as plt
import subprocess
from typing import List, Tuple
import os


def get_kmer_counts(sequence: str, k: int) -> Counter:
    """
    Count k-mers in a single DNA sequence.
    """
    counts = Counter()
    seq = sequence.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        counts[kmer] += 1
    return counts

def get_kmer_frequencies(sequences: list[str], k: int) -> dict[str, float]:
    """
    Compute normalized k-mer frequency distribution across multiple sequences.
    """
    total_counts = Counter()
    for seq in sequences:
        total_counts.update(get_kmer_counts(seq, k))
    total = sum(total_counts.values())
    return {kmer: count / total for kmer, count in total_counts.items()}

def kl_divergence(p: dict[str, float], q: dict[str, float], base: float = 2) -> float:
    """
    Compute KL-divergence KL(p || q) with small smoothing for zero entries.
    """
    eps = 1e-12
    all_keys = set(p) | set(q)
    kl = 0.0
    for key in all_keys:
        pv = p.get(key, eps)
        qv = q.get(key, eps)
        kl += pv * math.log(pv / qv, base)
    return kl

def js_divergence(p: dict[str, float], q: dict[str, float], base: float = 2) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    """
    # Construct midpoint distribution m = (p + q) / 2
    all_keys = set(p) | set(q)
    m = {key: (p.get(key, 0.0) + q.get(key, 0.0)) / 2 for key in all_keys}
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)

# Example usage (replace with your real sequences list):
# real_seqs = ["ATGCGT...", ...]
# gen_seqs = ["ATGCGT...", ...]
# real_freq = get_kmer_frequencies(real_seqs, k=4)
# gen_freq  = get_kmer_frequencies(gen_seqs, k=4)
# divergence = js_divergence(real_freq, gen_freq)
# print("4-mer JS divergence:", divergence)


# --- GC content and sequence length analysis ---

def compute_gc_content(sequences: list[str]) -> list[float]:
    """
    Compute GC content (fraction of G and C bases) for each sequence.
    Returns a list of GC fractions (0.0–1.0).
    """
    gc_contents = []
    for seq in sequences:
        seq = seq.upper()
        if len(seq) == 0:
            gc_contents.append(0.0)
        else:
            gc = seq.count('G') + seq.count('C')
            gc_contents.append(gc / len(seq))
    return gc_contents

def compute_sequence_lengths(sequences: list[str]) -> list[int]:
    """
    Compute the length of each sequence.
    Returns a list of integer lengths.
    """
    return [len(seq) for seq in sequences]



def plot_gc_distribution(real_gcs: list[float],
                         base_gcs: list[float],
                         ft_gcs:   list[float],
                         bins:     int = 50):
    """
    Plot overlaid histograms of GC-content distributions for:
    - real sequences
    - base-generated sequences
    - fine-tuned-generated sequences
    using proportions instead of raw counts.
    """
    plt.figure()
    plt.hist(real_gcs, bins=bins, density=True, alpha=0.5, label='Real')
    plt.hist(base_gcs, bins=bins, density=True, alpha=0.5, label='Base-generated')
    plt.hist(ft_gcs,   bins=bins, density=True, alpha=0.5, label='Fine-tuned')
    plt.xlabel('GC content')
    plt.ylabel('Proportion')
    plt.legend()
    plt.title('GC-content Distribution')
    plt.show()

def plot_length_distribution(real_lengths: list[int],
                             base_lengths: list[int],
                             ft_lengths:   list[int],
                             bins:         int = 50):
    """
    Plot overlaid histograms of sequence length distributions for:
    - real sequences
    - base-generated sequences
    - fine-tuned-generated sequences
    using proportions instead of raw counts.
    """
    plt.figure()
    plt.hist(real_lengths, bins=bins, density=True, alpha=0.5, label='Real')
    plt.hist(base_lengths, bins=bins, density=True, alpha=0.5, label='Base-generated')
    plt.hist(ft_lengths,   bins=bins, density=True, alpha=0.5, label='Fine-tuned')
    plt.xlabel('Sequence length')
    plt.ylabel('Proportion')
    plt.legend()
    plt.title('Sequence Length Distribution')
    plt.show()


# --- Similarity and divergence metrics ---

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # Initialization
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compute_min_edit_distances(gen_seqs: List[str], real_seqs: List[str]) -> List[int]:
    """
    For each generated sequence, compute the minimum Levenshtein distance
    to any sequence in the real_seqs list.
    """
    min_distances = []
    for gen in gen_seqs:
        # Compute distance to each real seq and take minimum
        distances = [levenshtein_distance(gen, real) for real in real_seqs]
        min_distances.append(min(distances) if distances else None)
    return min_distances

def plot_edit_distance_distribution(real_dists: List[int],
                                    base_dists: List[int],
                                    ft_dists:   List[int],
                                    bins:       int = 50):
    """
    Plot overlaid histograms of minimum edit-distance distributions.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(real_dists, bins=bins, density=True, alpha=0.5, label='Real (self-dist)')
    plt.hist(base_dists, bins=bins, density=True, alpha=0.5, label='Base-generated')
    plt.hist(ft_dists,   bins=bins, density=True, alpha=0.5, label='Fine-tuned')
    plt.xlabel('Min Levenshtein Distance to Real')
    plt.ylabel('Proportion')
    plt.legend()
    plt.title('Novelty: Edit-Distance Distribution')
    plt.show()


def _read_fasta_sequences(path: str) -> list[str]:
    seqs: list[str] = []
    p = os.path.abspath(path)
    if os.path.isdir(p):
        for fname in os.listdir(p):
            if not fname.lower().endswith((".fa", ".fasta", ".fna", ".fas")):
                continue
            seqs.extend(_read_fasta_sequences(os.path.join(p, fname)))
        return seqs
    with open(p, "r") as fh:
        buf = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            seqs.append("".join(buf))
    return seqs


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="K-mer and sequence stats analysis.")
    ap.add_argument("--real-fasta")
    ap.add_argument("--base-fasta")
    ap.add_argument("--ft-fasta")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    if not (args.real_fasta or args.base_fasta or args.ft_fasta):
        return

    real = _read_fasta_sequences(args.real_fasta) if args.real_fasta else []
    base = _read_fasta_sequences(args.base_fasta) if args.base_fasta else []
    ft = _read_fasta_sequences(args.ft_fasta) if args.ft_fasta else []

    out = {"k": args.k}
    if real:
        out["real"] = {
            "count": len(real),
            "gc_mean": sum(compute_gc_content(real)) / len(real),
            "len_mean": sum(compute_sequence_lengths(real)) / len(real),
        }
    if base:
        out["base"] = {
            "count": len(base),
            "gc_mean": sum(compute_gc_content(base)) / len(base),
            "len_mean": sum(compute_sequence_lengths(base)) / len(base),
        }
    if ft:
        out["ft"] = {
            "count": len(ft),
            "gc_mean": sum(compute_gc_content(ft)) / len(ft),
            "len_mean": sum(compute_sequence_lengths(ft)) / len(ft),
        }

    if real and base:
        out["js_real_vs_base"] = js_divergence(
            get_kmer_frequencies(real, args.k),
            get_kmer_frequencies(base, args.k),
        )
    if real and ft:
        out["js_real_vs_ft"] = js_divergence(
            get_kmer_frequencies(real, args.k),
            get_kmer_frequencies(ft, args.k),
        )

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

import os
import glob

def load_fasta_seqs(directory):
    """
    Load all .fasta and .fa sequences from `directory` into a list of strings.
    """
    seqs = []
    for pattern in ("*.fasta", "*.fa"):
        for fp in glob.glob(os.path.join(directory, pattern)):
            with open(fp, "r") as fh:
                # skip header lines, join the sequence lines
                seq = "".join(line.strip() 
                              for line in fh 
                              if not line.startswith(">"))
                seqs.append(seq)
    return seqs

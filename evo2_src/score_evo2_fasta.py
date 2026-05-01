import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from evo2 import Evo2


def read_fasta(path: str):
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq).upper()
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            yield name, "".join(seq).upper()


def mean_logprob_per_token(evo2_model, seq: str) -> float:
    ids = evo2_model.tokenizer.tokenize(seq)  # list[int]
    device = next(evo2_model.model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = evo2_model.model(input_ids)

    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_logp = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return float(token_logp.mean().cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_name", default="evo2_1b_base")
    args = ap.parse_args()

    evo2 = Evo2(args.model_name)
    # put model in eval mode + ensure on GPU if available
    evo2.model.eval()

    rows = []
    for name, seq in read_fasta(args.fasta):
        score = mean_logprob_per_token(evo2, seq)
        rows.append((name, len(seq), score))
        print(name, len(seq), score)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "length_bp", "mean_logprob_per_token"])
        w.writerows(rows)

    scores = np.array([r[2] for r in rows], dtype=float)
    print("N =", len(scores))
    print("Mean =", scores.mean())
    print("Std =", scores.std())


if __name__ == "__main__":
    main()
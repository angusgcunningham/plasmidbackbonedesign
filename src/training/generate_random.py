#!/usr/bin/env python
import json, os, re, random, torch
from transformers import PreTrainedTokenizerFast
from datetime import datetime

# ── USER SETTINGS ───────────────────────────────────────────────────────
SEQUENCE_NUMBER = 1000
START_SEQUENCE   = (
'ATG'
)

TARGET_MEAN_BP  = 4000
TARGET_SPREAD_BP= 1500
TEMPERATURE     = 1.05
TOP_P           = 0.95
REPETITION_PEN  = 1.05
NO_REPEAT_NGRAM = None
MAX_NEW_TOK_CAP = 12000

TOKENIZER_FILE  = "/cs/student/projects1/aibh/2024/acunning/Projects/models/addgene_trained_dna_tokenizer.json"
OUTPUT_DIR      = "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_random_atg"

# ---- utils ----
def sample_target_bp(mean=TARGET_MEAN_BP, spread=TARGET_SPREAD_BP):
    lo, hi = max(500, mean - spread), mean + spread
    return int(random.triangular(lo, hi, mean))

def strip_to_acgtn(s: str) -> str:
    return re.sub(r"[^ACGTNacgtn]", "", s).upper()

def cut_circular_to_target(bases: str, target_bp: int, seed_len=64, min_frac=0.6, max_frac=1.4):
    if len(bases) <= target_bp:
        return bases
    if len(bases) < seed_len + 1:
        return bases[:target_bp]
    seed = bases[:seed_len]
    min_gap = int(target_bp * min_frac)
    max_len = int(target_bp * max_frac)
    j = bases.find(seed, min_gap)
    if j != -1 and j <= max_len:
        return bases[:j]
    return bases[:min(max_len, max(target_bp, 1000))]

def build_bp_len_table(tokenizer):
    # precompute bp length contributed by each token id (counts A/C/G/T/N only)
    vocab_size = tokenizer.vocab_size
    bp_per_tok = []
    for i in range(vocab_size):
        s = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        bp_per_tok.append(sum(ch in "ACGTNacgtn" for ch in s))
    return torch.tensor(bp_per_tok, dtype=torch.long)


def _get_banned_ngram_tokens(prev_ids, no_repeat_ngram_size):
    if no_repeat_ngram_size is None or no_repeat_ngram_size <= 1:
        return set()
    if len(prev_ids) < no_repeat_ngram_size - 1:
        return set()

    # Build mapping: prefix -> set(next_tokens)
    n = no_repeat_ngram_size
    ngrams = {}
    for i in range(len(prev_ids) - n + 1):
        prefix = tuple(prev_ids[i : i + n - 1])
        token = prev_ids[i + n - 1]
        ngrams.setdefault(prefix, set()).add(token)

    current_prefix = tuple(prev_ids[-(n - 1):])
    return ngrams.get(current_prefix, set())


def _sample_next_token(vocab_size, device, top_p, temperature, banned, valid_mask):
    # Random logits -> random tokens baseline, but keep sampling strategy (top_p, temperature).
    logits = torch.rand(vocab_size, device=device)

    if temperature and temperature != 1.0:
        logits = logits / float(temperature)

    if banned:
        banned_idx = torch.tensor(list(banned), device=device, dtype=torch.long)
        logits[banned_idx] = float("-inf")

    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask, float("-inf"))

    if top_p is not None and top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        if not torch.any(keep):
            keep[0] = True
        mask = torch.ones_like(keep, dtype=torch.bool)
        mask[keep] = False
        remove_idx = sorted_idx[mask]
        logits[remove_idx] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_and_save(
    tokenizer,
    seed_ids,
    device,
    output_dir,
    generations_fasta,
    metadata_csv,
):
    os.makedirs(output_dir, exist_ok=True)

    # Precompute table once
    bp_per_tok = build_bp_len_table(tokenizer).to(device)
    vocab_size = bp_per_tok.numel()
    prompt_len = seed_ids.shape[1]

    # Only sample from tokens that contribute at least 1 bp
    valid_mask = (bp_per_tok > 0)

    # --- before the loop in generate_and_save() ---
    os.makedirs(output_dir, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"gen_run_{run_id}.txt")

    header = {
        "run_id": run_id,
        "model": "random_vocab_baseline",
        "tokenizer": TOKENIZER_FILE if "TOKENIZER_FILE" in globals() else None,
        "seed_len_bp": len(START_SEQUENCE),
        "settings": {
            "target_mean_bp": TARGET_MEAN_BP if "TARGET_MEAN_BP" in globals() else None,
            "target_spread_bp": TARGET_SPREAD_BP if "TARGET_SPREAD_BP" in globals() else None,
            "temperature": TEMPERATURE,
            "top_p": TOP_P if "TOP_P" in globals() else 1.0,
            "repetition_penalty": REPETITION_PEN if "REPETITION_PEN" in globals() else 1.0,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM if "NO_REPEAT_NGRAM" in globals() else None,
            "max_new_tokens_cap": MAX_NEW_TOK_CAP if "MAX_NEW_TOK_CAP" in globals() else None,
            "sequence_number": SEQUENCE_NUMBER,
        },
        "columns": ["idx", "target_bp", "raw_bp", "final_bp", "trimmed", "outfile"]
    }
    with open(log_path, "w") as fh:
        fh.write(json.dumps(header, indent=2) + "\n")
        fh.write("\t".join(header["columns"]) + "\n")

    if not os.path.exists(metadata_csv):
        with open(metadata_csv, "w") as fh:
            fh.write("plasmid_id,target_bp,raw_bp,final_bp,trimmed,outfile\n")

    for i in range(SEQUENCE_NUMBER):
        target_bp = sample_target_bp()

        generated_ids = []
        bp_so_far = 0
        steps = 0
        while bp_so_far < target_bp and steps < MAX_NEW_TOK_CAP:
            banned = _get_banned_ngram_tokens(generated_ids, NO_REPEAT_NGRAM)
            tok = _sample_next_token(
                vocab_size=vocab_size,
                device=device,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                banned=banned,
                valid_mask=valid_mask,
            )
            generated_ids.append(tok)
            bp_so_far += int(bp_per_tok[tok].item())
            steps += 1

        out_ids = torch.cat([seed_ids[0], torch.tensor(generated_ids, device=device, dtype=torch.long)])
        txt   = tokenizer.decode(out_ids, skip_special_tokens=True)
        bases = strip_to_acgtn(txt)
        final = cut_circular_to_target(bases, target_bp)

        trimmed = int(len(final) < len(bases))
        outname = f"random_generated_{i:03d}.fasta"
        with open(log_path, "a") as fh:
            fh.write(f"{i+1}\t{target_bp}\t{len(bases)}\t{len(final)}\t{trimmed}\t{outname}\n")

        print(f"[{i+1}/{SEQUENCE_NUMBER}] target≈{target_bp} bp | generated={len(final)} bp")
        plasmid_id = f"RandomGPT_generate{i}"
        with open(os.path.join(output_dir, outname), "w") as fh:
            fh.write(f">{plasmid_id}\n{final}\n")
        with open(generations_fasta, "a") as fh:
            fh.write(f">{plasmid_id}\n{final}\n")
        with open(metadata_csv, "a") as fh:
            fh.write(f"{plasmid_id},{target_bp},{len(bases)},{len(final)},{trimmed},{outname}\n")


def main() -> None:
    import argparse
    from datetime import datetime
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Generate plasmid sequences by random token baseline.")
    ap.add_argument("--tokenizer-json", default=TOKENIZER_FILE)
    ap.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--generations-fasta", default=None)
    ap.add_argument("--metadata-csv", default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    args = ap.parse_args()

    if args.num_samples is not None:
        global SEQUENCE_NUMBER
        SEQUENCE_NUMBER = args.num_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_json)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = "[PAD]"
    else:
        tokenizer.pad_token = tokenizer.eos_token

    run_dir = Path("runs") / args.run_name
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "generations"
    generations_fasta = args.generations_fasta or str(output_dir / "generations.fasta")
    metadata_csv = args.metadata_csv or str(output_dir / "generations_metadata.csv")
    Path(generations_fasta).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_csv).parent.mkdir(parents=True, exist_ok=True)

    seed_ids = tokenizer.encode(START_SEQUENCE.upper(), return_tensors="pt", add_special_tokens=False).to(device)
    generate_and_save(
        tokenizer,
        seed_ids,
        device,
        str(output_dir),
        generations_fasta,
        metadata_csv,
    )


if __name__ == "__main__":
    main()

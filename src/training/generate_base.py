# generate_base2.py
#!/usr/bin/env python
import json, os, re, random, torch
from transformers import (
    GPT2LMHeadModel, PreTrainedTokenizerFast, GPT2Config,
    StoppingCriteria, StoppingCriteriaList
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from datetime import datetime

# ── USER SETTINGS ───────────────────────────────────────────────────────
SEQUENCE_NUMBER   = 1000
START_SEQUENCE   = (
'ATG'
)

TARGET_MEAN_BP    = 4000        # aim around 4 kb
TARGET_SPREAD_BP  = 1500        # ± range
TEMPERATURE       = 1.05
TOP_P             = 0.95         # nucleus sampling to avoid greedy copies
REPETITION_PEN    = 1.05        # gentle anti-copy bias - keep small
MAX_NEW_TOK_CAP   = 12000       # safety cap on *tokens*
OUTPUT_DIR        = "<PATH_TO_OUTPUT>"

# ── PATHS ───────────────────────────────────────────────────────────────
BASE_PT        = "pretrained_model.pt"
TOKENIZER_JSON = "addgene_trained_dna_tokenizer.json"

# ── Helpers ─────────────────────────────────────────────────────────────
def patch_gpt2_attention(model):
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.config = model.config
            if not hasattr(module, "_attn_implementation"):
                module._attn_implementation = None

def load_base_model(pt_path, device):
    raw = torch.load(pt_path, weights_only=False, map_location="cpu")
    state = raw.state_dict() if hasattr(raw, "state_dict") else raw
    cfg = getattr(raw, "config", GPT2Config())
    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(state, strict=False)
    patch_gpt2_attention(model)
    model.config.use_cache = True
    return model.to(device).eval()

def sample_target_bp(mean=TARGET_MEAN_BP, spread=TARGET_SPREAD_BP):
    lo, hi = max(500, mean - spread), mean + spread
    return int(random.triangular(lo, hi, mean))

def strip_to_acgtn(s: str) -> str:
    return re.sub(r"[^ACGTNacgtn]", "", s).upper()

def load_fasta_sequence(path: str) -> str:
    seq = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return strip_to_acgtn("".join(seq))

def cut_circular_to_target(bases: str, target_bp: int, seed_len=64, min_frac=0.6, max_frac=1.4):
    """Prefer to stop where the starting seed reappears (circular closure)."""
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
    """Precompute how many A/C/G/T/N each token contributes."""
    bp_per_tok = []
    for i in range(tokenizer.vocab_size):
        s = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        bp_per_tok.append(sum(ch in "ACGTNacgtn" for ch in s))
    return torch.tensor(bp_per_tok, dtype=torch.long)

class BasePairLimit(StoppingCriteria):
    def __init__(self, bp_per_tok: torch.LongTensor, prompt_len: int, target_bp: int):
        self.bp_per_tok = bp_per_tok
        self.prompt_len = prompt_len
        self.target_bp  = target_bp
    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        new_tok_ids = seq[self.prompt_len:]
        bp_so_far = int(self.bp_per_tok[new_tok_ids].sum().item()) if new_tok_ids else 0
        return bp_so_far >= self.target_bp

@torch.no_grad()
def generate_and_save(
    model,
    tokenizer,
    seed_ids,
    seed_label,
    seed_len_bp,
    device,
    output_dir,
    generations_fasta,
    metadata_csv,
    run_name,
):
    os.makedirs(output_dir, exist_ok=True)

    # Precompute once
    bp_per_tok = build_bp_len_table(tokenizer).to(device)
    os.makedirs(output_dir, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"gen_run_{run_id}.txt")
    
    # header with all decode + length params
    header = {
        "run_id": run_id,
        "model": FT_MODEL_DIR if "FT_MODEL_DIR" in globals() else BASE_PT,
        "tokenizer": TOKENIZER_FILE if "TOKENIZER_FILE" in globals() else TOKENIZER_JSON,
        "seed_label": seed_label,
        "seed_len_bp": seed_len_bp,
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
        prompt_len = seed_ids.shape[1]
        target_bp = sample_target_bp()
        stop_crit = StoppingCriteriaList([BasePairLimit(bp_per_tok, prompt_len, target_bp)])

        out = model.generate(
            input_ids=seed_ids,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PEN,
            max_new_tokens=MAX_NEW_TOK_CAP,      # safety
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
            stopping_criteria=stop_crit,
        )[0]

        txt   = tokenizer.decode(out, skip_special_tokens=True)
        bases = strip_to_acgtn(txt)
        final = cut_circular_to_target(bases, target_bp)

        trimmed = int(len(final) < len(bases))   # 1 if circular/target trim applied
        outname = f"{run_name}_{i+1:04d}.fasta"
        with open(log_path, "a") as fh:
            fh.write(f"{i+1}\t{target_bp}\t{len(bases)}\t{len(final)}\t{trimmed}\t{outname}\n")

        print(f"[{i+1}/{SEQUENCE_NUMBER}] target≈{target_bp} bp | generated={len(final)} bp")
        plasmid_id = f"{run_name}_{i+1:04d}"
        with open(os.path.join(output_dir, outname), "w") as fh:
            fh.write(f">{plasmid_id}\n{final}\n")
        with open(generations_fasta, "a") as fh:
            fh.write(f">{plasmid_id}\n{final}\n")
        with open(metadata_csv, "a") as fh:
            fh.write(f"{plasmid_id},{target_bp},{len(bases)},{len(final)},{trimmed},{outname}\n")

        if device.type == "cuda":
            torch.cuda.empty_cache()

def main() -> None:
    import argparse
    from datetime import datetime
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Generate plasmid sequences from base model.")
    ap.add_argument("--base-pt", default=BASE_PT)
    ap.add_argument("--tokenizer-json", default=TOKENIZER_JSON)
    ap.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--preset", default="base")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--generations-fasta", default=None)
    ap.add_argument("--metadata-csv", default=None)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--seed", default=START_SEQUENCE)
    ap.add_argument("--seed-choice", choices=["ATG", "GFP", "both"], default="ATG")
    ap.add_argument("--seed-fasta", default=None, help="FASTA file for GFP seed.")
    args = ap.parse_args()

    if args.num_samples is not None:
        global SEQUENCE_NUMBER
        SEQUENCE_NUMBER = args.num_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_json)
    # Ensure a pad token; keep EOS as-is if present
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = "[PAD]"
    else:
        tokenizer.pad_token = tokenizer.eos_token

    run_dir = Path("runs") / args.run_name
    base_output_dir = Path(args.output_dir) if args.output_dir else run_dir / "generations"

    model_base = load_base_model(args.base_pt, device)
    model_base.resize_token_embeddings(len(tokenizer))
    model_base.config.vocab_size = len(tokenizer)

    seed_atg = args.seed or START_SEQUENCE
    seed_choice = args.seed_choice
    seed_labels = []
    seed_seqs = []
    if seed_choice in ("GFP", "both"):
        if not args.seed_fasta:
            raise ValueError("--seed-fasta is required when seed-choice is GFP or both.")
        gfp_seed = load_fasta_sequence(args.seed_fasta)
    else:
        gfp_seed = None
    if seed_choice in ("ATG", "both"):
        seed_labels.append("ATG")
        seed_seqs.append(seed_atg)
    if seed_choice in ("GFP", "both"):
        seed_labels.append("GFP")
        seed_seqs.append(gfp_seed)
    seed_ids_list = [
        tokenizer.encode(
            seq.upper(),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        for seq in seed_seqs
    ]
    seed_lengths = [len(seq) for seq in seed_seqs]

    for seed_ids, seed_label, seed_len in zip(seed_ids_list, seed_labels, seed_lengths):
        out_dir = base_output_dir / f"gen_{args.run_name}_{seed_label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        generations_fasta = str(out_dir / "generations.fasta")
        metadata_csv = str(out_dir / "generations_metadata.csv")
        generate_and_save(
            model_base,
            tokenizer,
            seed_ids,
            seed_label,
            seed_len,
            device,
            str(out_dir),
            generations_fasta,
            metadata_csv,
            args.run_name,
        )
    print(f"✅ Done. Check {base_output_dir}")


if __name__ == "__main__":
    main()

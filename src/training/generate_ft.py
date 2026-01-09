#!/usr/bin/env python
import json, os, re, random, torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, StoppingCriteria, StoppingCriteriaList
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

FT_MODEL_DIR    = "/cs/student/projects1/aibh/2024/acunning/Projects/models/plasmidgpt-finetuned-35k-stride-1024"
TOKENIZER_FILE  = "/cs/student/projects1/aibh/2024/acunning/Projects/models/addgene_trained_dna_tokenizer.json"
OUTPUT_DIR      = "/cs/student/projects1/aibh/2024/acunning/Projects/Results/generated_seqs/gen_ft35_atg"

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

class BasePairLimit(StoppingCriteria):
    def __init__(self, bp_per_tok: torch.LongTensor, prompt_len: int, target_bp: int):
        self.bp_per_tok = bp_per_tok
        self.prompt_len = prompt_len
        self.target_bp  = target_bp
    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        # single sequence generation => batch size 1
        seq = input_ids[0].tolist()
        new_tok_ids = seq[self.prompt_len:]
        bp_so_far = int(self.bp_per_tok[new_tok_ids].sum().item()) if new_tok_ids else 0
        return bp_so_far >= self.target_bp

@torch.no_grad()
def generate_and_save(
    model,
    tokenizer,
    seed_ids,
    device,
    output_dir,
    generations_fasta,
    metadata_csv,
):
    os.makedirs(output_dir, exist_ok=True)
    model.to(device).eval()
    model.config.use_cache = True

    # Precompute table once
    bp_per_tok = build_bp_len_table(tokenizer).to(device)
    prompt_len = seed_ids.shape[1]
    
    # --- before the loop in generate_and_save() ---
    os.makedirs(output_dir, exist_ok=True)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"gen_run_{run_id}.txt")
    
    # write a header with all decode + length params
    header = {
        "run_id": run_id,
        "model": FT_MODEL_DIR if "FT_MODEL_DIR" in globals() else BASE_PT,
        "tokenizer": TOKENIZER_FILE if "TOKENIZER_FILE" in globals() else TOKENIZER_JSON,
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
        stop_crit = StoppingCriteriaList([BasePairLimit(bp_per_tok, prompt_len, target_bp)])

        out = model.generate(
            input_ids=seed_ids,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PEN,
            max_new_tokens=MAX_NEW_TOK_CAP,  # safety
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
            stopping_criteria=stop_crit,
        )[0]

        txt   = tokenizer.decode(out, skip_special_tokens=True)
        bases = strip_to_acgtn(txt)
        final = cut_circular_to_target(bases, target_bp)

        trimmed = int(len(final) < len(bases))   # 1 if circular/target trim applied
        outname = f"ft_generated_{i:03d}.fasta"
        with open(log_path, "a") as fh:
            fh.write(f"{i+1}\t{target_bp}\t{len(bases)}\t{len(final)}\t{trimmed}\t{outname}\n")


        print(f"[{i+1}/{SEQUENCE_NUMBER}] target≈{target_bp} bp | generated={len(final)} bp")
        plasmid_id = f"PlasmidGPT_generate{i}"
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

    ap = argparse.ArgumentParser(description="Generate plasmid sequences from fine-tuned model.")
    ap.add_argument("--ft-model-dir", default=FT_MODEL_DIR)
    ap.add_argument("--tokenizer-json", default=TOKENIZER_FILE)
    ap.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--preset", default="ft")
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

    model = GPT2LMHeadModel.from_pretrained(args.ft_model_dir, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)

    seed_ids = tokenizer.encode(START_SEQUENCE.upper(), return_tensors="pt", add_special_tokens=False).to(device)
    generate_and_save(
        model,
        tokenizer,
        seed_ids,
        device,
        str(output_dir),
        generations_fasta,
        metadata_csv,
    )


if __name__ == "__main__":
    main()

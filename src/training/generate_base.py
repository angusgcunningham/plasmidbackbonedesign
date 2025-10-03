# generate_base_loop.py

#!/usr/bin/env python
import torch
import os

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

# ─── USER SETTINGS ────────────────────────────────────────────────────
SEQUENCE_NUMBER  = 1000
START_SEQUENCE   = (
'ATG'
)

MIN_EXTRA_TOKENS = 1
MAX_LENGTH       = 1000
TEMPERATURE      = 1.0
DO_SAMPLE        = True

# ─── PATHS ─────────────────────────────────────────────────────────────
BASE_PT        = "/cs/student/projects1/aibh/2024/acunning/Projects/models/pretrained_model.pt"
TOKENIZER_JSON = "/cs/student/projects1/aibh/2024/acunning/Projects/models/addgene_trained_dna_tokenizer.json"
OUTPUT_DIR    = "/cs/student/projects1/aibh/2024/acunning/Projects/Results/gen_base_atg"

def patch_gpt2_attention(model):
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.config = model.config
            if not hasattr(module, "_attn_implementation"):
                module._attn_implementation = None

def load_base_model(pt_path, device):
    raw = torch.load(pt_path, weights_only = False, map_location="cpu")
    state = raw.state_dict() if hasattr(raw, "state_dict") else raw
    cfg = getattr(raw, "config", GPT2Config())
    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(state, strict=False)
    patch_gpt2_attention(model)
    model.config.use_cache = False
    torch.cuda.empty_cache()
    return model.to(device)

def generate_and_save(model, tokenizer, input_ids, device):
    model.eval()
    model.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    for i in range(SEQUENCE_NUMBER):
        while True:
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask         = attention_mask,
                    max_length             = MAX_LENGTH,
                    do_sample              = DO_SAMPLE,
                    temperature            = TEMPERATURE,
                    top_k                  = 0,
                    top_p                  = 1.0,
                    pad_token_id           = tokenizer.eos_token_id,
                    eos_token_id           = tokenizer.eos_token_id,
                    num_return_sequences   = 1,
                )[0]
            seq = tokenizer.decode(out, skip_special_tokens=True).replace(" ", "")
            if len(seq) > len(START_SEQUENCE) + MIN_EXTRA_TOKENS:
                break

        print(f"[{i+1}/{SEQUENCE_NUMBER}] Generated length={len(seq)}")
        fname = f"base_generated_{i:03d}.fasta"
        filepath = os.path.join(OUTPUT_DIR, fname)
        with open(filepath, "w") as fh:
            fh.write(f">PlasmidGPT_generate{i}\n{seq}\n")

        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load tokenizer and special tokens
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[PROMPT]", "[PROMPT2]"]
    })

    # Load and patch base model
    model_base = load_base_model(BASE_PT, device)
    model_base.resize_token_embeddings(len(tokenizer))
    model_base.config.vocab_size = len(tokenizer)

    # Prepare input_ids
    seed_ids = tokenizer.encode(
        START_SEQUENCE.upper(),
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    p_tok  = tokenizer.convert_tokens_to_ids("[PROMPT]")
    p2_tok = tokenizer.convert_tokens_to_ids("[PROMPT2]")
    special = torch.tensor([p_tok]*10 + [p2_tok], device=device)
    input_ids = torch.cat((special.unsqueeze(0), seed_ids), dim=1)

    # Generate sequences
    generate_and_save(model_base, tokenizer, input_ids, device)

    print(" Done. Check {OUTPUT_DIR}/base_generated_sequence_*.fasta")
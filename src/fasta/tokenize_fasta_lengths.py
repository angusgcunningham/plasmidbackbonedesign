from pathlib import Path
from Bio import SeqIO
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import math, csv

# --- Paths / params ---
fasta_dir = Path("/cs/student/projects1/aibh/2024/acunning/Projects/Data/dataset/fasta")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="/cs/student/projects1/aibh/2024/acunning/Projects/models/addgene_trained_dna_tokenizer.json"
)
CTX, STRIDE = 2048, 1024
out_csv = "/cs/student/projects1/aibh/2024/acunning/Projects/Data/token_stats.csv"

# Ensure pad token exists (matches your training setup)
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"
else:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load records like your script ---
examples = []
for fn in fasta_dir.glob("*.fa*"):
    for rec in SeqIO.parse(str(fn), "fasta"):
        seq = str(rec.seq).upper()
        examples.append({"file": fn.name, "record_id": rec.id, "text": seq})

ds = Dataset.from_list(examples)

def windows_count(n_tok, ctx=CTX, stride=STRIDE):
    if n_tok <= ctx:
        return 1
    # number of starting positions: 1 (at 0) + ceil((n-ctx)/stride)
    return 1 + math.ceil((n_tok - ctx) / stride)

def add_stats(batch):
    # Use add_special_tokens=False to measure *pure sequence* length;
    # switch to True if your training adds BOS/EOS per window.
    encs = tokenizer(batch["text"], add_special_tokens=False)
    lens = [len(ids) for ids in encs["input_ids"]]
    wins = [windows_count(n) for n in lens]
    return {"tok_len": lens, f"windows_ctx{CTX}_stride{STRIDE}": wins}

ds = ds.map(add_stats, batched=True)

# --- Print quick summary ---
tok_lens = ds["tok_len"]
tok_lens_sorted = sorted(tok_lens)
p50 = tok_lens_sorted[len(tok_lens_sorted)//2]
p95 = tok_lens_sorted[int(0.95*len(tok_lens_sorted))-1]
over_ctx = sum(1 for x in tok_lens if x > CTX)
total_windows = sum(ds[f"windows_ctx{CTX}_stride{STRIDE}"])

print(f"Sequences: {len(ds)}")
print(f"Token length: min={tok_lens_sorted[0]}, p50={p50}, p95={p95}, max={tok_lens_sorted[-1]}")
print(f">{CTX} tokens: {over_ctx} ({over_ctx/len(ds):.1%})")
print(f"Estimated windows (CTX={CTX}, stride={STRIDE}): {total_windows}")


import matplotlib.pyplot as plt

# 1) Compute token lengths per sequence (same as training: specials included)
enc = tokenizer(ds["text"], add_special_tokens=True)
tok_lens = [len(ids) for ids in enc["input_ids"]]

# 2) Histogram of token lengths
plt.figure()
plt.hist(tok_lens, bins=100)
plt.xlabel("Tokenized length per sequence")
plt.ylabel("Count")
plt.title("Distribution of tokenized sequence lengths")
plt.show()

# 3) % of padded tokens given CTX/STRIDE (matches your sliding window loop)
CTX, STRIDE = 2048, 1024
def windows_and_pads(n, ctx=CTX, stride=STRIDE):
    if n <= ctx:
        return 1, ctx - n
    # number of windows from range(0, n, stride)
    n_win = math.ceil(n / stride)
    last_start = (n_win - 1) * stride
    last_len = max(0, n - last_start)
    pad_last = max(0, ctx - last_len) if last_len < ctx else 0
    return n_win, pad_last

total_tokens = 0
total_pads = 0
for n in tok_lens:
    n_win, pad_last = windows_and_pads(n)
    total_tokens += n_win * CTX
    total_pads   += pad_last

pad_pct = 100.0 * total_pads / max(1, total_tokens)
print(f"Estimated padded tokens across all windows: {pad_pct:.2f}%")

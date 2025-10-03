import torch, gc
# delete any lingering references
for name in ["model","trainer","tok_ds","data_collator"]:
    if name in globals(): del globals()[name]
gc.collect()
torch.cuda.empty_cache()




from pathlib          import Path
from Bio               import SeqIO
from datasets          import Dataset
from transformers      import PreTrainedTokenizerFast, DataCollatorWithPadding, Trainer, TrainingArguments, GenerationConfig
import torch
import random

# Read  FASTA into a list of dicts 
fasta_dir = Path("<PATH TO FASTA FILES>")
examples = []
for fn in fasta_dir.glob("*.fa*"):
    for rec in SeqIO.parse(str(fn), "fasta"):
        # Uppercase 
        seq = str(rec.seq).upper()
        examples.append({"text": seq})

print(f"Loaded {len(examples)} sequences.")

# Build a HF Dataset 
ds = Dataset.from_list(examples)

# Context window

CTX = 2048

tokenizer = PreTrainedTokenizerFast(tokenizer_file="PATH TO TOKENIZER JSON FILE")

# Ensure padding token with no eos_token
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"
else:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize FULL sequences - no windowing, no pre-padding
def tokenize_full(batch):
    enc = tokenizer(batch["text"], add_special_tokens=True)
    ids = enc["input_ids"]
    return {"input_ids": ids, "length": [len(x) for x in ids]}

tok_ds = ds.map(tokenize_full, batched=True, remove_columns=["text"])

# Quick check
lens = sorted(tok_ds["length"])
print(f"Tokenized seqs: {len(tok_ds)} | min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")

# Collator: one 2048-token view per plasmid with circular crop
_base_pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

def clm_collator_one_window(batch):
    cropped = []
    for ex in batch:
        ids = ex["input_ids"]
        n = len(ids)
        if n <= CTX:
            cropped.append(ids)  # short: keep whole; base collator will pad within-batch
        else:
            # circular crop in token space
            start = random.randint(0, n - 1)
            ids2 = ids + ids  # allow wraparound
            cropped.append(ids2[start:start + CTX])
    # dynamic padding to max length in batch
    b = _base_pad([{"input_ids": x} for x in cropped])
    labels = b["input_ids"].clone()
    labels[b["attention_mask"] == 0] = -100  # ignore pads in loss
    b["labels"] = labels
    return b

data_collator = clm_collator_one_window
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model object directly
model = torch.load(
    "<PATH TO MODEL FILE>",
    map_location=device,
    weights_only=False     
).to(device)
'''
if device=="cuda":
    model = model.half()
    model.gradient_checkpointing_enable()
'''



from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

for module in model.modules():
    # 1) Give every GPT2Attention a .config pointing to the model’s config
    if isinstance(module, GPT2Attention):
        module.config = model.config
    # 2) Ensure every module has an _attn_implementation attribute
    if not hasattr(module, "_attn_implementation"):
        module._attn_implementation = None


class DummyGenConfig:
    def __getattr__(self, name):
        return None

# Attach the dummy to model
model.generation_config = DummyGenConfig()

# Verify no blow ups:
print("min_p:", model.generation_config.min_p)
print("anything_else:", model.generation_config.foo_bar)


# Disable cache 
model.config.use_cache = False

# Attach GenerationConfig (so Trainer’s eval/logging never hits a missing attribute)
model.generation_config = GenerationConfig()

# Quick error check:
print("use_cache:", model.config.use_cache)
print("min_p attribute:", getattr(model.generation_config, "min_p"))
print("dummy gen config is callable?", callable(model.generation_config))

# TrainingArguments & Trainer
args = TrainingArguments(
    output_dir="ft-plasmidgpt-15k",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    gradient_checkpointing=True,
    fp16=False,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=500,
    group_by_length=True,             # bucket by 'length'
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,             # full sequences
    data_collator=data_collator,      # random circular crop → one window/plasmid/step
)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
print("Model device:", next(model.parameters()).device)
print("➤ Model parameter dtype:", next(model.parameters()).dtype)
if torch.cuda.is_available():
    print(torch.cuda.memory_summary(device=0, abbreviated=True))

#  Train 
trainer.train()

#  Save 
trainer.save_model("plasmidgpt-finetuned-15k-complete")
tokenizer.save_pretrained("plasmidgpt-finetuned-15k-complete")

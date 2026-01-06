import torch, gc
# delete any lingering references
for name in ["model","trainer","tok_ds","data_collator"]:
    if name in globals(): del globals()[name]
gc.collect()
torch.cuda.empty_cache()




from pathlib          import Path
from Bio               import SeqIO
from datasets          import Dataset
from transformers      import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# Read every FASTA into a list of dicts 
fasta_dir = Path("<PATH TO FASTA FILES>")
examples = []
for fn in fasta_dir.glob("*.fa*"):
    for rec in SeqIO.parse(str(fn), "fasta"):
        # Uppercase
        seq = str(rec.seq).upper()
        examples.append({"text": seq})

print(f"Loaded {len(examples)} sequences.")

#  Build a HF Dataset
ds = Dataset.from_list(examples)

#  Tokenizer stride

CTX = 2048
STRIDE = 1024  

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/cs/student/projects1/aibh/2024/acunning/Projects/models/addgene_trained_dna_tokenizer.json")

# Ensure padding token with no eos_token
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = "[PAD]"
else:
    tokenizer.pad_token = tokenizer.eos_token

def sliding_window_tokenize(batch):
    all_input_ids = []
    for text in batch["text"]:
        enc = tokenizer(text, return_attention_mask=False, return_tensors="pt")["input_ids"].squeeze(0)
        total_len = enc.size(0)

        # Slide over the input using a window of CTX and stride of STRIDE
        for i in range(0, total_len, STRIDE):
            window = enc[i: i + CTX]
            if window.size(0) < CTX:
                pad_len = CTX - window.size(0)
                window = torch.cat([window, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            all_input_ids.append(window)

    return {"input_ids": all_input_ids}

# Apply to dataset
tok_ds = ds.map(sliding_window_tokenize, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")

# Check number of windows
print(f"→ {len(tok_ds)} windows of {CTX} tokens each")
print(f"→ {sum(len(x) for x in tok_ds['input_ids'])} total tokens")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#  Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model object directly
model = torch.load(
    "<PATH TO MODEL FILE>",
    map_location=device,
    weights_only=False   
)
model = model.to(device)
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



#model.gradient_checkpointing_enable()

#  Patch out missing GenerationConfig attrs 
# Some Trainer internals probe model.generation_config.min_p (and friends).
# We’ll replace it with a dummy object that returns None for any attribute.

class DummyGenConfig:
    def __getattr__(self, name):
        return None

# Attach the dummy to model
model.generation_config = DummyGenConfig()

# Verify no blows up:
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

# TrainingArguments & Trainer ─
args = TrainingArguments(
    output_dir="ft-plasmidgpt-20k",
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
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    data_collator=data_collator,
)


print("CUDA available:", torch.cuda.is_available())        # should now be True
print("CUDA device count:", torch.cuda.device_count())      # should be ≥ 1
print("Current device:", torch.cuda.current_device())       # e.g. 0
print("Model device:", next(model.parameters()).device)

# Check dtype of model parameters
dtype = next(model.parameters()).dtype
print("➤ Model parameter dtype:", dtype)

# 2) Check if gradient checkpointing is on
#is_ckpt = model.is_gradient_checkpointing_enabled()
#print("➤ Gradient checkpointing enabled:", is_ckpt)

# Look at current GPU memory usage
print(torch.cuda.memory_summary(device=0, abbreviated=True))

def check_gpu_memory(device=0):
    print(f"--- GPU Memory on device {device} ---")
    print(torch.cuda.memory_summary(device=device, abbreviated=True))
    print("--------------------------------------")

# Call this function after key steps:
check_gpu_memory()


#  Train
trainer.train()


#  Save 
trainer.save_model("plasmidgpt-finetuned-20k-complete")
tokenizer.save_pretrained("plasmidgpt-finetuned-20k-complete")


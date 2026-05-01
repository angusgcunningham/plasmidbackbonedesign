import argparse
import gc
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import torch
import wandb
from Bio import SeqIO
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    TrainerCallback,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Avoid tokenizers fork/parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _sanitize_value(val):
    if isinstance(val, torch.dtype):
        return str(val).replace("torch.", ""), True, "dtype"
    if callable(val):
        name = getattr(val, "__name__", None)
        return name if isinstance(name, str) else repr(val), True, "callable"
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val, False, ""
    if isinstance(val, dict):
        changed = False
        out = {}
        for k, v in val.items():
            nv, ch, _ = _sanitize_value(v)
            out[k] = nv
            changed = changed or ch
        return out, changed, "dict"
    if isinstance(val, (list, tuple)):
        changed = isinstance(val, tuple)
        out = []
        for v in val:
            nv, ch, _ = _sanitize_value(v)
            out.append(nv)
            changed = changed or ch
        return out, changed, "list"
    return repr(val), True, "repr"


def sanitize_config(cfg) -> list[str]:
    sanitized = []
    for key, val in list(vars(cfg).items()):
        new_val, changed, _ = _sanitize_value(val)
        if changed:
            sanitized.append(f"{key}({type(val).__name__})")
            setattr(cfg, key, new_val)
        else:
            try:
                json.dumps(val)
            except TypeError:
                setattr(cfg, key, repr(val))
                sanitized.append(f"{key}({type(val).__name__})")
    return sanitized


def _stringify_dtypes(obj):
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if callable(obj):
        name = getattr(obj, "__name__", None)
        return name if isinstance(name, str) else repr(obj)
    if isinstance(obj, dict):
        return {k: _stringify_dtypes(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_dtypes(v) for v in obj]
    return obj


def _sanitize_config_inplace(cfg) -> None:
    if cfg is None:
        return
    for key in list(cfg.__dict__.keys()):
        val = cfg.__dict__.get(key)
        if key.startswith("to_") or callable(val):
            del cfg.__dict__[key]
            continue
        cfg.__dict__[key] = _stringify_dtypes(val)


class SanitizeConfigCallback(TrainerCallback):
    def __init__(self):
        self._reported = False

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        _sanitize_config_inplace(getattr(model, "config", None))
        _sanitize_config_inplace(getattr(model, "generation_config", None))
        if not self._reported:
            print("[config] sanitized on save")
            self._reported = True
        return control


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune GPT-2 on plasmid FASTA (perplexity eval, full-seq).")
    ap.add_argument("--fasta-dir", default="<PATH TO FASTA FILES>")
    ap.add_argument("--tokenizer-json", default="PATH TO TOKENIZER JSON FILE")
    ap.add_argument("--model-path", default="<PATH TO MODEL FILE>")
    ap.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--preset", default="ft35k_perp")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--num-epochs", type=float, default=3.0)
    ap.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "200")))
    args = ap.parse_args()

    # delete any lingering references
    for name in ["model", "trainer", "tok_ds", "data_collator"]:
        if name in globals():
            del globals()[name]
    gc.collect()
    torch.cuda.empty_cache()

    # Read FASTA into a list of dicts
    fasta_dir = Path(args.fasta_dir)
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

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_json)

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
        args.model_path,
        map_location=device,
        weights_only=False
    ).to(device)
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)

    if old_vocab_size != new_vocab_size:
        print(f"Resizing embeddings from {old_vocab_size} → {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)
    else:
        print("Embedding size already matches tokenizer.")

    print("Final embedding size:", model.get_input_embeddings().weight.shape[0])
    # Optional mixed-precision / checkpointing tweaks (kept commented on purpose)
    # if device == "cuda":
    #     model = model.half()
    #     model.gradient_checkpointing_enable()

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
    # Some older GPT2Config objects lack internal _output_attentions/_output_hidden_states.
    # Set them explicitly to avoid AttributeError in forward.
    if not hasattr(model.config, "_output_attentions"):
        model.config._output_attentions = False
    if not hasattr(model.config, "_output_hidden_states"):
        model.config._output_hidden_states = False
    if not hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "eager"

    # Attach GenerationConfig (so Trainer’s eval/logging never hits a missing attribute)
    model.generation_config = GenerationConfig()
    sanitize_config(model.config)
    sanitize_config(model.generation_config)
    _sanitize_config_inplace(model.config)
    _sanitize_config_inplace(model.generation_config)

    # Quick error check:
    print("use_cache:", model.config.use_cache)
    print("min_p attribute:", getattr(model.generation_config, "min_p"))
    print("dummy gen config is callable?", callable(model.generation_config))

    # Create a 95/5 train/test split for perplexity evaluation
    split = tok_ds.train_test_split(test_size=0.05, seed=42, shuffle=True)
    train_ds = split["train"]
    test_ds = split["test"]

    run_dir = Path("runs") / args.run_name
    save_dir = Path(args.save_dir) if args.save_dir else run_dir / "models" / args.preset
    output_dir = Path(args.output_dir) if args.output_dir else save_dir / "trainer"

    # TrainingArguments & Trainer
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=False,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=500,
        group_by_length=True,             # bucket by 'length'
        dataloader_num_workers=2,
        report_to="wandb",
        run_name=os.environ.get("WANDB_RUN_NAME", args.run_name),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,           # full sequences
        eval_dataset=test_ds,             # 5% perplexity test set
        data_collator=data_collator,      # random circular crop → one window/plasmid/step
        callbacks=[SanitizeConfigCallback()],
    )

    class PeriodicEvalCallback(TrainerCallback):
        def __init__(self, trainer, eval_steps):
            self.trainer = trainer
            self.eval_steps = eval_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % self.eval_steps == 0:
                metrics = self.trainer.evaluate(eval_dataset=test_ds)
                if "eval_loss" in metrics and metrics["eval_loss"] is not None:
                    try:
                        ppl = math.exp(metrics["eval_loss"])
                    except OverflowError:
                        ppl = float("inf")
                    wandb.log({"eval/perplexity": ppl})
            return control

    trainer.add_callback(PeriodicEvalCallback(trainer, args.eval_steps))

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
    print("Model device:", next(model.parameters()).device)
    print("➤ Model parameter dtype:", next(model.parameters()).dtype)
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary(device=0, abbreviated=True))

    # Train
    trainer.train()

    # Evaluate perplexity on the 5% test split
    metrics = trainer.evaluate(eval_dataset=test_ds)
    if "eval_loss" in metrics and metrics["eval_loss"] is not None:
        try:
            ppl = math.exp(metrics["eval_loss"])
        except OverflowError:
            ppl = float("inf")
        print(f"Perplexity on 5% test set: {ppl}")
        wandb.log({"eval/perplexity": ppl})

    # Save
    trainer.save_model(str(save_dir))
    if model.generation_config is not None:
        model.generation_config.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))


if __name__ == "__main__":
    main()

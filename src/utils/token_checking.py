import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast

# This script performs diagnostic checks to verify that the new masked tokenizer is properly aligned with the model's embeddings after fine-tuning. It checks:
# 1. That the new tokenizer's vocab size matches the model's embedding size.
# 2. That specific tokens have the same IDs in the new tokenizer as they did in the original tokenizer (for unchanged tokens).
# 3. That tokenization of a test sequence results in more tokens with the new tokenizer (indicating that macro-tokens were successfully split).
# 4. That the model's embeddings are accessible for tokens in the new tokenizer.

# ===============================
# 🔧 EDIT THESE PATHS
# ===============================

# === PATHS ===
ORIGINAL_TOKENIZER_JSON = "/home/acunningham/Projects/plasmidbackbonedesign/plasmidbackbonedesign/models/addgene_trained_dna_tokenizer.json"

NEW_TOKENIZER_JSON = "/home/acunningham/Projects/plasmidbackbonedesign/plasmidbackbonedesign/models/2-ft15kplasmidgpt-maxtok50-tested-complete/tokenizer.json"

MODEL_PATH = "/home/acunningham/Projects/plasmidbackbonedesign/plasmidbackbonedesign/models/2-ft15kplasmidgpt-maxtok50-tested-complete/"

# === LOAD TOKENIZERS ===
old_tokenizer = PreTrainedTokenizerFast(tokenizer_file=ORIGINAL_TOKENIZER_JSON)
new_tokenizer = PreTrainedTokenizerFast(tokenizer_file=NEW_TOKENIZER_JSON)

# === LOAD MODEL ===
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

print("Loaded everything successfully.")


# ===============================
# 1️⃣ CHECK VOCAB SIZE VS EMBEDDINGS
# ===============================

print("==== VOCAB / EMBEDDING SIZE CHECK ====")
print("New tokenizer vocab size:", len(new_tokenizer))
print("Model embedding size:", model.get_input_embeddings().weight.shape[0])

if len(new_tokenizer) == model.get_input_embeddings().weight.shape[0]:
    print("✅ Embedding size matches tokenizer vocab size.\n")
else:
    print("❌ Embedding size DOES NOT match tokenizer vocab size.\n")

# ===============================
# 2️⃣ CHECK TOKEN ID ALIGNMENT
# ===============================

print("==== TOKEN ID ALIGNMENT CHECK ====")

test_tokens = ["ATG", "GFP", "TATA", "CAT", "bla", "ori"]

for tok in test_tokens:
    old_id = old_tokenizer.convert_tokens_to_ids(tok)
    new_id = new_tokenizer.convert_tokens_to_ids(tok)

    print(f"Token: {tok}")
    print(f"  Old ID: {old_id}")
    print(f"  New ID: {new_id}")

    if old_id != new_id:
        print("  ⚠️  ID CHANGED")
    else:
        print("  ✅ ID SAME")
    print()

print("NOTE:")
print("- If many IDs changed AND you loaded pretrained weights,")
print("  then embeddings were initially misaligned (scrambled).")
print()

# ===============================
# 3️⃣ CHECK TOKEN COUNT CHANGE
# ===============================

print("==== TOKENIZATION LENGTH CHECK ====")

test_sequence = (
    "ATGCGTATCGATCGATCGATCGTACGATCGATCGTACGATCGATCGATCGTACGATCGATCGT"
)

old_tokens = old_tokenizer.tokenize(test_sequence)
new_tokens = new_tokenizer.tokenize(test_sequence)

print("Old tokenizer token count:", len(old_tokens))
print("New tokenizer token count:", len(new_tokens))

if len(new_tokens) > len(old_tokens):
    print("✅ New tokenizer is using smaller tokens (expected).")
else:
    print("⚠️ Token count did not increase much — macro-tokens may not have been common.")
print()

# ===============================
# 4️⃣ CHECK EMBEDDING SANITY
# ===============================

print("==== EMBEDDING SANITY CHECK ====")

token = "ATG"
token_id = new_tokenizer.convert_tokens_to_ids(token)

if token_id is not None and token_id >= 0:
    embedding_vector = model.get_input_embeddings().weight[token_id]
    print(f"Embedding vector (first 5 values) for '{token}':")
    print(embedding_vector[:5])
    print("\n(Just confirming embeddings are accessible.)")
else:
    print(f"Token '{token}' not found in new tokenizer.")

print("\n==== DIAGNOSTIC COMPLETE ====\n")
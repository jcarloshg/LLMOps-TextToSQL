import os
from unsloth import FastLanguageModel

# 1. LOAD THE FINE-TUNED MODEL
print("Loading the trained LoRA adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # The folder we saved in Phase 3
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 2. EXPORT TO GGUF
# This automatically merges the LoRA and quantizes the model.
# "q4_k_m" is the recommended 4-bit quantization method.
print("Exporting to GGUF format... (This will take 5-10 minutes)")
model.save_pretrained_gguf(
    "exported_model",
    tokenizer,
    quantization_method="q4_k_m"
)

print("Export Complete! Check the 'exported_model' folder.")

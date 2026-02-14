import json
import os
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# 1. MLOPS SETUP: Connect to our Dockerized MLflow server
# ============================================================================
# MLflow tracks experiments, metrics, and model artifacts during training
# This environment variable tells MLflow which experiment to log results to
os.environ["MLFLOW_EXPERIMENT_NAME"] = "qwen-sql-finetune"

# ============================================================================
# 2. LOAD & FORMAT DATA
# ============================================================================
# Qwen2.5-Coder expects prompts in ChatML format (conversation structure).
# We convert raw question-SQL pairs into ChatML format:
#   - <|im_start|>system: Sets context (Postgres expertise, schema info)
#   - <|im_start|>user: The natural language question
#   - <|im_start|>assistant: The expected SQL answer
# This structure helps the model learn the mapping between English and SQL

print("Loading synthetic data...")
with open("data/raw_sql_pairs.json", "r") as f:
    raw_data = json.load(f)

# Convert each question-SQL pair into ChatML format
formatted_data = []
for item in raw_data:
    prompt = (
        f"<|im_start|>system\nYou are a Postgres SQL expert. Use the company_sales_2024 schema.<|im_end|>\n"
        f"<|im_start|>user\n{item['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n{item['sql']}<|im_end|>"
    )
    formatted_data.append({"text": prompt})

# Convert list of dictionaries into a Hugging Face Dataset object
dataset = Dataset.from_list(formatted_data)

# ============================================================================
# 3. LOAD MODEL & APPLY LORA
# ============================================================================
# We use Qwen2.5-Coder-1.5B, a 1.5-billion parameter model optimized for SQL tasks (fits in 6GB GPU).
# Unsloth provides memory-optimized inference & training for faster GPU training.

max_seq_length = 512  # Maximum input/output token length (reduced for GPU memory)
print("Loading base model via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B",  # Smaller model for GPU memory constraints
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,  # 4-bit quantization: reduces memory footprint by 75%
)

# LoRA (Low-Rank Adaptation): Instead of training all 7B parameters,
# we only train small adapter weights (~2% of parameters). This is much faster
# and uses less GPU memory while achieving comparable fine-tuning results.
# We target attention and feed-forward layers where most learning happens.
model = FastLanguageModel.get_peft_model(
    model,
    # Rank of the low-rank matrices (larger = more capacity, more memory)
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj",  # Attention layers
                    "o_proj", "gate_proj", "up_proj", "down_proj"],  # Feed-forward
    lora_alpha=32,  # Scaling factor for LoRA updates
    lora_dropout=0,  # No dropout in LoRA layers
    bias="none",  # Don't train bias parameters
    use_gradient_checkpointing="unsloth",  # Save memory during backprop
    random_state=3407,  # Fixed seed for reproducibility
)

# ============================================================================
# 4. CONFIGURE TRAINER
# ============================================================================
# SFTTrainer (Supervised Fine-Tuning Trainer) handles the training loop.
# It automatically computes loss based on predicting SQL given questions.

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Field containing the ChatML prompts
    max_seq_length=max_seq_length,
    dataset_num_proc=2,  # Use 2 CPU processes for data loading
    args=TrainingArguments(
        # Batch size per GPU (reduced for memory constraints)
        per_device_train_batch_size=1,
        # Accumulate gradients over 4 steps for larger effective batch
        gradient_accumulation_steps=4,
        warmup_steps=5,  # Gradually increase learning rate for first 5 steps
        # Total training steps (keep low for lab; increase for production)
        max_steps=60,
        learning_rate=2e-4,  # Learning rate for LoRA parameters
        # Use bf16 for RTX 4050
        bf16=True,
        fp16=False,
        logging_steps=1,  # Log metrics every step
        output_dir="outputs",  # Directory to save checkpoints
        report_to=[],  # Disabled: MLflow not available in trainer container
    ),
)

# ============================================================================
# 5. EXECUTE TRAINING
# ============================================================================
# The trainer runs the fine-tuning loop:
#   - For each batch: predict SQL tokens given question tokens
#   - Compute loss (difference between predicted and actual SQL)
#   - Backpropagate through LoRA layers only
#   - Update LoRA weights using optimizer
# Progress and metrics are logged to MLflow at http://localhost:5000

print("Starting Fine-Tuning... Check MLflow at http://localhost:5000")
trainer.train()

# ============================================================================
# 6. SAVE ARTIFACTS
# ============================================================================
# Save the trained LoRA adapter weights and tokenizer.
# These are much smaller than the full 7B model (~100MB vs 14GB).
# During inference, we load the base model + these adapters to get the fine-tuned behavior.

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Saved LoRA adapter to /lora_model")

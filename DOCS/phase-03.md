This is where we cross the line from standard software development into true AI Engineering.

In this phase, we treat model weights as trackable artifacts, utilizing container orchestration principles to ensure reproducibility. We will use **Unsloth** to fine-tune the model (which requires an NVIDIA GPU) and **MLflow** to track our experiment parameters and loss curves.

Here is the step-by-step guide for **Phase 3: Customization (Fine-Tuning)**.

| Section            | Details                                                                                                                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase Name**     | **Phase 3: Customization (Fine-Tuning)**                                                                                                                                                                                                                        |
| **Description**    | Adapting the base model to your specific database schema using Low-Rank Adaptation (LoRA). Instead of just running a script, we track training metrics (like loss) using an observability server to ensure the model is actually learning, not just memorizing. |
| **Key Activities** | • Formatting the synthetic data into the model's expected prompt structure (ChatML).<br>                                                                                                                                                                        |

<br>• Configuring LoRA hyperparameters (Rank, Alpha).<br>

<br>• Running the training loop inside a GPU-accelerated container.<br>

<br>• Logging experiment metrics to MLflow. |
| **Tools** | • **Unsloth:** The standard for local fine-tuning. It uses custom kernels to train 2x faster and use 60% less VRAM.<br>

<br>• **MLflow:** An open-source platform for managing the ML lifecycle and visualizing training graphs. |
| **Prerequisites** | An **NVIDIA GPU** and the `nvidia-container-toolkit` installed on your host machine to allow Docker to access the GPU. |
| **Tips & Best Practices** | • **Log Everything:** Always track your hyperparameter changes in MLflow. "Vibes" don't scale; logged data does.<br>

<br>• **Start Small:** Run your training for just 50-60 steps first to ensure the pipeline doesn't crash before committing to a multi-hour training run. |
| **Learning Resources** | **[Unsloth Official Docker Guide](https://unsloth.ai/docs/get-started/install/docker)** – Documentation on the exact container we are using today. |

---

### **Step 1: Update the Docker Architecture**

We need to add two new services to our `docker-compose.yml`: an MLflow tracking server (the "Ops" layer) and an Unsloth training container (the "Compute" layer).

Update your `docker-compose.yml` to include these:

```yaml
services:
  # ... (Keep your existing ollama-service from Phase 1)

  # Service 3: MLflow Tracking Server
  mlflow-server:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: sql_mlflow
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - ./mlruns:/mlruns # Persist experiment data

  # Service 4: Unsloth GPU Trainer
  unsloth-trainer:
    image: unsloth/unsloth:latest
    container_name: sql_unsloth_trainer
    volumes:
      - .:/workspace/work
    working_dir: /workspace/work
    # GPU Pass-through (Requires nvidia-container-toolkit on host)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    command: tail -f /dev/null
```

Start the new infrastructure:

```bash
docker compose up -d

```

You can now open your browser to `http://localhost:5000` to see your empty MLflow dashboard.

---

### **Step 2: The Training Script**

We will write a script that loads your JSON data, formats it into a conversation template, and trains the LoRA adapter while automatically streaming metrics to MLflow.

Create `src/train.py`:

```python
import json
import os
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. MLOPS SETUP: Connect to our Dockerized MLflow server
os.environ["MLFLOW_EXPERIMENT_NAME"] = "qwen-sql-finetune"

# 2. LOAD & FORMAT DATA
# We must format our JSON into the "ChatML" structure Qwen expects
print("Loading synthetic data...")
with open("data/raw_sql_pairs.json", "r") as f:
    raw_data = json.load(f)

formatted_data = []
for item in raw_data:
    # Creating a structured prompt: System Context -> User Request -> AI Answer
    prompt = (
        f"<|im_start|>system\nYou are a Postgres SQL expert. Use the company_sales_2024 schema.<|im_end|>\n"
        f"<|im_start|>user\n{item['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n{item['sql']}<|im_end|>"
    )
    formatted_data.append({"text": prompt})

dataset = Dataset.from_list(formatted_data)

# 3. LOAD MODEL & APPLY LORA
max_seq_length = 2048
print("Loading base model via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B", # Unsloth's optimized weights
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True, # 4-bit Quantization saves massive VRAM
)

# Apply LoRA (We only train ~2% of the model's parameters)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. CONFIGURE TRAINER
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60, # Keep it short for the lab. Increase for real training.
        learning_rate=2e-4,
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        bf16=FastLanguageModel.is_bfloat16_supported(),
        logging_steps=1,
        output_dir="outputs",
        report_to="mlflow", # <- The crucial MLOps integration
    ),
)

# 5. EXECUTE TRAINING
print("Starting Fine-Tuning... Check MLflow at http://localhost:5000")
trainer.train()

# 6. SAVE ARTIFACTS
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Saved LoRA adapter to /lora_model")

```

---

### **Step 3: Execute and Monitor**

Execute the training script from inside the GPU-enabled Unsloth container:

```bash
docker exec -it sql_unsloth_trainer python src/train.py

```

**The "Ops" Verification:**
While the script is running, go to `http://localhost:5000` in your browser.

1. Click on the `qwen-sql-finetune` experiment.
2. Click on the active run.
3. Check the **Metrics** section. You should see a graph for `train/loss`. If the loss curve is trending downward, your model is successfully learning your SQL schema!

Once the script finishes, you will see a new folder in your repository called `lora_model/`. This folder contains the specialized weights (the adapter) that know your database.

**Ready for Phase 4?**
In Phase 4, we will write an automated test script to evaluate whether this new LoRA model actually writes better SQL than the baseline model we tested in Phase 1. Shall we proceed to the Evaluation phase?

[Running Unsloth in a Docker Container for LLM Training](https://www.youtube.com/watch?v=nCxLzpju_eE)
This video provides a complete visual walkthrough of launching the Unsloth image with GPU pass-through to ensure your host system remains clean during the fine-tuning process.

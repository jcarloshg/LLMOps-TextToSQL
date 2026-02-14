This is the final hurdle before your model becomes a real product.

In Phase 4, we evaluated the model to ensure it was accurate. However, raw LoRA weights sitting in a folder aren't useful for a web application. In **Phase 5**, we will merge your LoRA adapter into the base model, compress it using **GGUF quantization**, and deploy it as a standalone, production-ready API using Ollama.

Here is your step-by-step guide for **Phase 5: Deployment**.

| Section            | Details                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Phase Name**     | **Phase 5: Deployment & Serving**                                                                                                                                              |
| **Description**    | Transforming the fine-tuned weights into a highly optimized, deployable artifact. We compress the model so it runs fast and cheap in production, then serve it via a REST API. |
| **Key Activities** | • Merging the LoRA adapter back into the Qwen base model.<br>                                                                                                                  |

<br>• Quantizing the model to 4-bit GGUF format for faster inference.<br>

<br>• Creating a declarative `Modelfile` (similar to a Dockerfile).<br>

<br>• Serving the model via Ollama's API. |
| **Tools** | • **GGUF (llama.cpp):** The industry standard file format for running local LLMs efficiently.<br>

<br>• **Ollama Modelfile:** To configure the model's system prompt and parameters at the server level. |
| **Prerequisites** | The `lora_model/` folder generated from your Unsloth training in Phase 3. |
| **Tips & Best Practices** | • **Use Q4_K_M Quantization:** This offers the best balance. It shrinks a 7B model from 14GB down to ~4GB with almost zero drop in SQL accuracy.<br>

<br>• **Bake in the Prompt:** Put your System Prompt directly into the Modelfile so front-end developers don't have to remember it. |
| **Learning Resources** | **[Ollama Modelfile Documentation](https://docs.ollama.com/modelfile)** – Learn how to configure advanced parameters like context windows and stop tokens. |

---

### **Step 1: Update Docker Volumes**

Currently, our `ollama-service` container cannot see the files in your project directory. We need to mount your workspace so Ollama can access the GGUF file we are about to create.

Open your `docker-compose.yml` and update the `ollama-service` to include a new volume:

```yaml
ollama-service:
  image: ollama/ollama:latest
  container_name: sql_model_server
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
    - .:/app # <-- ADD THIS LINE to share your local folder with Ollama
  working_dir: /app
```

Run this command to apply the changes:

```bash
docker compose up -d

```

---

### **Step 2: Export and Quantize to GGUF**

We will use Unsloth's built-in exporter. This script will take your base model, fuse your LoRA adapter into it, and compress it into a single `.gguf` file.

Create `src/export_model.py`:

```python
import os
from unsloth import FastLanguageModel

# 1. LOAD THE FINE-TUNED MODEL
print("Loading the trained LoRA adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model", # The folder we saved in Phase 3
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

```

Run this script inside your GPU-enabled Unsloth container:

```bash
docker exec -it sql_unsloth_trainer python src/export_model.py

```

_Note: When this finishes, you will see a new `.gguf` file inside the `exported_model/` folder. It will likely be named `exported_model-unsloth.Q4_K_M.gguf`._

---

### **Step 3: Create the Ollama Modelfile**

Just as Docker uses a `Dockerfile` to build images, Ollama uses a `Modelfile` to build AI models. We will bake your exact requirements into the model itself.

Create a file named `Modelfile` in your root directory:

```text
# 1. Point to the GGUF file you just exported
# (Update the filename below if Unsloth named it slightly differently)
FROM ./exported_model/exported_model-unsloth.Q4_K_M.gguf

# 2. Set the temperature to 0 for strict, deterministic code generation
PARAMETER temperature 0

# 3. Bake in the System Prompt so the API client doesn't have to send it
SYSTEM """You are a Postgres SQL expert.
Your only job is to generate SQL queries based on the company_sales_2024 schema.
Always divide total_arr_cents by 100 to get dollars.
Output ONLY valid SQL, with no markdown formatting or explanations."""

```

---

### **Step 4: Build and Deploy the Model**

Now we tell the Ollama server to ingest the Modelfile and spin up the production endpoint.

Execute this command inside the `ollama-service` container:

```bash
docker exec -it sql_model_server ollama create qwen-sql-prod -f Modelfile

```

Once it says "success", your custom model is officially deployed! You can verify it exists by running:

```bash
docker exec -it sql_model_server ollama list

```

_(You should see `qwen-sql-prod` sitting alongside the base `qwen2.5-coder` model)._

---

### **Phase 5 Verification: Test the Production API**

To prove your deployment works, we will hit the Ollama API directly, simulating how a web application (like a Node.js or Spring Boot backend) would communicate with your AI.

Run this simple `curl` command from your host terminal or `lab-runner`:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen-sql-prod",
  "prompt": "What is the total revenue for the APAC region?",
  "stream": false
}'

```

**Expected Outcome:**
You should receive a clean JSON response containing _only_ the SQL query, perfectly tailored to your custom database schema. No hallucinations, no generic chat responses—just pure SQL.

**Congratulations on deploying your custom AI!** You have successfully gone from generic weights to a specialized, containerized API.

The final missing piece of the LLMOps puzzle is **Phase 6: Monitoring**. Now that it's live, we need to track if the model starts failing in production using Arize Phoenix. Would you like the steps for this final phase?

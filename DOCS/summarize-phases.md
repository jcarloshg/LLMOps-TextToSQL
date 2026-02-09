# The "Text-to-SQL" Tuner (Fine-Tuning Focus)

**Lifecycle Emphasis:** Data Prep, Training, & Model Versioning.
**Why:** Generic models are bad at writing SQL for specific company schemas. You will fine-tune a small model (e.g., Llama-3-8B or Qwen-2.5-Coder) to master _your_ specific database schema.

| Phase                | Local Lab Implementation                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Selection**     | **Model:** `Llama-3-8B-Instruct` or `Qwen-2.5-Coder-7B`<br>**Reason:** Small enough to fine-tune on a single GPU (or CPU with LoRA).                                                                 |
| **2. Data Prep**     | **Task:** Generate 500 pairs of (Natural Language -> SQL Query) using a larger model (ChatGPT/Claude).<br>**Tool:** `DVC` (Data Version Control) to track dataset versions (`v1_raw`, `v2_cleaned`). |
| **3. Customization** | **Task:** Fine-Tune using **LoRA** (Low-Rank Adaptation).<br>**Tool:** **Unsloth** (runs 2x faster locally). Log training loss curves to **MLflow**.                                                 |
| **4. Evaluation**    | **Task:** "Execution Accuracy" – Run the generated SQL against a local SQLite DB. Does it crash? Does it return the right rows?<br>**Tool:** A simple Python test script that executes queries.      |
| **5. Deployment**    | **Task:** Convert the fine-tuned adapter to a `.gguf` file and load it into Ollama.<br>**Tool:** `Ollama Modelfile` (e.g., `FROM llama3`, `ADAPTER ./my_sql_lora.gguf`).                             |
| **6. Monitoring**    | **Task:** Track "Syntax Error Rate" in production.<br>**Tool:** **Arize Phoenix** (Trace: User Query -> Model -> SQL -> DB Error?).                                                                  |

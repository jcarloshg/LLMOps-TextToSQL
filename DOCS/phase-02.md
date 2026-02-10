# **Phase 2: Data Preparation & Versioning**

**Objective:**
Generate a high-quality "Golden Dataset" of 500+ (Question -> SQL) pairs derived from your specific schema, and version it using DVC so we can roll back if needed.

**Prerequisites:**
You need a "Teacher Model" to generate this data. Since this is a local lab, you can use:

- **Option A (Free/Local):** `Llama-3-8B` (via Ollama) with a very strong system prompt.
- **Option B (Paid/Better):** OpenAI/Claude API (better for generating complex synthetic data).
- _I will write the code to support **Ollama** by default so it remains free._

---

### **Step 1: Update Environment (Add DVC)**

We need to add DVC to our `lab-runner` container.

1. **Update `requirements.txt**`:
Add `dvc` to your existing file.

```text
langchain
langchain-community
langchain-ollama
dvc

```

2. **Rebuild the Container**:

```bash
docker compose up -d --build

```

3. **Enter the Lab Runner**:

```bash
docker exec -it sql_lab_runner bash

```

---

### **Step 2: Initialize DVC (Inside Container)**

Inside the container, we set up DVC. Since we don't have an S3 bucket for this local lab, we will use a local folder as our "remote" storage to simulate a real production setup.

```bash
# 1. Initialize Git (DVC requires git)
git init

# 2. Initialize DVC
dvc init

# 3. Create a "Local Remote" (Simulates S3 bucket on your hard drive)
mkdir -p /tmp/dvc-storage
dvc remote add -d mylocal /tmp/dvc-storage

# 4. Commit setup
git add .
git commit -m "Initialize DVC with local remote"

```

---

### **Step 3: The Generator Script**

We need a script that takes your schema and "brainstorms" questions and SQL queries.

**Create `src/generate_data.py`:**

````python
import json
import random
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP TEACHER MODEL
# We use a higher temperature to get diverse questions
llm = ChatOllama(
    base_url="http://ollama-service:11434",
    model="qwen2.5-coder:7b",
    temperature=0.7
)

# 2. DEFINE YOUR SCHEMA (The source of truth)
SCHEMA = """
Table: company_sales_2024
Columns:
- region_id (VARCHAR): 'NA', 'EMEA', 'APAC', 'LATAM'
- total_arr_cents (BIGINT): Revenue in cents. Divide by 100 for dollars.
- sales_rep_uid (UUID): Unique ID of the sales person.
- deal_closed_at (TIMESTAMP): When the deal was signed.
- product_tier (VARCHAR): 'Enterprise', 'SMB', 'Starter'
"""

# 3. THE PROMPT (Few-Shot Prompting for consistency)
system_prompt = """
You are a synthetic data generator. Your goal is to create training data for a Text-to-SQL model.
Given the schema below, generate 5 diverse SQL queries and their natural language questions.

Rules:
1. VARY difficulty: simple aggregations, date filtering, and string matching.
2. ALWAYS use the table name 'company_sales_2024'.
3. REMEMBER: 'total_arr_cents' is in cents.
4. Output strict JSON format list: [{"question": "...", "sql": "..."}, ...]

Schema:
{schema}
"""

prompt = ChatPromptTemplate.from_template(system_prompt)
chain = prompt | llm

def generate_batch(batch_size=5):
    print(f"Generating {batch_size} pairs...")
    try:
        response = chain.invoke({"schema": SCHEMA})
        # Extract JSON from potential markdown blocks
        content = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

# 4. MAIN LOOP
if __name__ == "__main__":
    dataset = []
    target_count = 50  # Start small for the lab, aim for 500 later

    while len(dataset) < target_count:
        batch = generate_batch()
        dataset.extend(batch)
        print(f"Total collected: {len(dataset)}")

    # Save to Raw Data Folder
    output_path = "data/raw_sql_pairs.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} pairs to {output_path}")

````

---

### **Step 4: Execute & Version Control**

Now we run the generation and "lock" the dataset version.

1. **Create Data Directory**:

```bash
mkdir data

```

2. **Run Generation**:

```bash
python src/generate_data.py

```

_(Wait for it to finish. You will see `data/raw_sql_pairs.json` appear.)_ 3. **Track with DVC**:
This is the "Ops" part. We don't commit the heavy JSON to git; we commit the `.dvc` pointer.

```bash
# Add file to DVC
dvc add data/raw_sql_pairs.json

# This creates 'data/raw_sql_pairs.json.dvc' (Metadata)
# and adds 'data/raw_sql_pairs.json' to .gitignore automatically

```

4. **Push to "Remote"**:

```bash
dvc push

```

5. **Git Commit the Metadata**:

```bash
git add data/raw_sql_pairs.json.dvc .gitignore
git commit -m "Add v1 of raw synthetic training data"
git tag v1_raw

```

---

### **Phase 2 Verification**

To confirm you have completed this phase correctly:

1. **Check Output:** Open `data/raw_sql_pairs.json`. Does it contain valid SQL?

- _Check:_ Does it divide cents by 100? If not, modify the prompt in `src/generate_data.py` and re-run.

2. **Check DVC:** Delete the JSON file (`rm data/raw_sql_pairs.json`).

- Run `dvc pull`.
- Did the file come back? If yes, your Data Ops pipeline is working.

**Ready for Phase 3?**
In Phase 3, we will use **Unsloth** (inside a GPU-enabled Docker container) to fine-tune the model using this JSON dataset.
_(Note: Phase 3 requires a GPU. If you don't have one, we will simulate training or use a CPU-friendly trainer like standard HuggingFace PEFT, though it will be slow.)_

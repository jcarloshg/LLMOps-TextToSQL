This is a fantastic project choice. **Text-to-SQL** is one of the highest-value problems in enterprise AI right now.

Here is the guide for **Phase 1: Model Selection & Baseline Setup**, completely Dockerized.

### **Phase 1: Model Selection & Environment Setup**

**Objective:**
We need to select the best "Base Model" for coding tasks and get it running inside a Docker container. We will establish a "Baseline" performance (how well it writes SQL _before_ fine-tuning).

**The Decision: `Qwen-2.5-Coder-7B**`
I recommend **Qwen-2.5-Coder-7B** over Llama-3-8B for this specific use case.

- **Why?** It is currently the State-of-the-Art (SOTA) open-weight model for code generation. It outperforms models 5x its size in SQL tasks.
- **Size:** 7 Billion parameters (fits easily on 16GB RAM or a consumer GPU).

---

### **Step 1: Project Structure**

Create a clean directory for your lab.

```bash
mkdir text-to-sql-lab
cd text-to-sql-lab
mkdir src
touch docker-compose.yml

```

---

### **Step 2: The Docker Architecture**

We will use **Docker Compose** to orchestrate two services:

1. **`ollama-service`**: The inference server hosting the model.
2. **`lab-runner`**: A Python environment where you will run your evaluation scripts and interactions.

Create `docker-compose.yml`:

```yaml
services:
  # Service 1: The Model Server (Ollama)
  ollama-service:
    image: ollama/ollama:latest
    container_name: sql_model_server
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama # Persist models so you don't re-download them
    # Uncomment the deploy section below if you have an NVIDIA GPU & Toolkit installed
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  # Service 2: The Client/Lab Environment
  lab-runner:
    image: python:3.10-slim
    container_name: sql_lab_runner
    volumes:
      - .:/app
    working_dir: /app
    command: tail -f /dev/null # Keep container running indefinitely
    environment:
      - OLLAMA_HOST=http://ollama-service:11434

volumes:
  ollama_data:
```

---

### **Step 3: Initialize the Environment**

1. **Start the containers:**

```bash
docker compose up -d

```

2. **Enter the model server to pull the model:**
   We need to download the `qwen2.5-coder:7b` weights into the `ollama-service` container.

```bash
docker exec -it sql_model_server ollama pull qwen2.5-coder:7b

```

_(Note: This is ~4.7GB. Wait for it to finish.)_ 3. **Setup the Client Environment:**
Now, let's set up the dependencies in your `lab-runner`.
Create a `requirements.txt` file in your root folder:

```text
langchain
langchain-community
langchain-ollama

```

Install them inside the running container:

```bash
docker exec sql_lab_runner pip install -r requirements.txt

```

---

### **Step 4: The Baseline Test (Sanity Check)**

Before we fine-tune (Phase 3), we must prove _why_ we need to. We will ask the base model to write SQL for a made-up company schema. It will likely fail or hallucinate table names because it doesn't know _your_ schema yet.

Create `src/baseline_check.py`:

```python
from langchain_ollama import ChatOllama

# Connect to the Dockerized Ollama service
llm = ChatOllama(
    base_url="http://ollama-service:11434",
    model="qwen2.5-coder:7b",
    temperature=0
)

# A specific, custom schema that the generic model CANNOT know.
schema_context = """
Table: company_sales_2024
Columns:
- region_id (VARCHAR)
- total_arr_cents (BIGINT) -- Annual Recurring Revenue in cents
- sales_rep_uid (UUID)
- deal_closed_at (TIMESTAMP)
"""

query = "Calculate the total revenue in dollars for the 'North' region in 2024."

prompt = f"""
You are a SQL expert. Write a Postgres SQL query for the following request.
Use ONLY the provided schema.

Schema:
{schema_context}

Request: {query}
"""

print(f"--- Asking Qwen-2.5-Coder (Base Model) ---\n")
response = llm.invoke(prompt)
print(response.content)

```

---

### **Step 5: Run the Verification**

Run the script inside your Docker container:

```bash
docker exec sql_lab_runner python src/baseline_check.py

```

### **Expected Outcome (Pass/Fail)**

- **Success:** You see a SQL query printed in your terminal.
- **The "Base Model" Problem:** Look closely at the output.
- Did it divide `total_arr_cents` by 100 to get dollars? (Generic models often miss specific column logic like "cents").
- Did it use the correct table name `company_sales_2024`?

**Why this is "Phase 1 Complete":**

1. **Infrastructure:** You have a Dockerized loop for running the LLM (`ollama-service`) and running code (`lab-runner`).
2. **Selection:** You have validated that `qwen2.5-coder:7b` runs on your hardware.
3. **Baseline:** You have a script to test the model "off the shelf."

**Ready for Phase 2?**
In Phase 2, we will generate the **Synthetic Data** needed to teach the model that `total_arr_cents` must always be divided by 100. Shall we proceed?

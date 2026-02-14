This is where we separate the prototypes from production-ready systems.

When evaluating Text-to-SQL models, comparing the generated text string to a "Gold" SQL string (Exact Match) is deeply flawed because there are many valid ways to write the same query. The industry standard is **Execution Accuracy (EX)**: executing both the AI-generated query and the ground-truth query against a real database and comparing the resulting datasets.

Since you have experience configuring PostgreSQL environments, we will skip the toy SQLite examples and use a fully containerized PostgreSQL database to run our execution tests.

Here is the step-by-step guide for **Phase 4: Evaluation**.

| Section            | Details                                                                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Phase Name**     | **Phase 4: Evaluation (Execution Accuracy)**                                                                                                                                                                                   |
| **Description**    | This phase proves the fine-tuned model's functional correctness. We execute the model's generated SQL and the known "Ground Truth" SQL against an actual database. If the returned dataframes match exactly, the model passes. |
| **Key Activities** | • Spinning up a sandboxed testing database.<br>                                                                                                                                                                                |

<br>• Seeding the database with mock data.<br>

<br>• Running batch inference with the fine-tuned LoRA model.<br>

<br>• Comparing result sets (rows and columns) programmatically. |
| **Tools** | • **PostgreSQL (Dockerized):** The real-world execution environment.<br>

<br>• **Pandas:** For easy dataframe comparison and handling unordered results.<br>

<br>• **Pytest:** To run the evaluation as a standard CI/CD testing suite. |
| **Prerequisites** | Understanding of database connections (connection strings, ports) and basic Python data manipulation. |
| **Tips & Best Practices** | • **Read-Only Users:** Always evaluate LLM-generated SQL using a database user with strictly read-only permissions to prevent accidental `DROP TABLE` hallucinations.<br>

<br>• **Order Agnostic:** Use dataframe sorting before comparison. `SELECT a, b` and `SELECT b, a` might be functionally equivalent depending on the business need. |
| **Learning Resources** | **[Defog.ai SQL-Eval](https://github.com/defog-ai/sql-eval)** – An excellent open-source repository demonstrating how to handle edge cases in execution-based SQL evaluation. |

---

### **Step 1: Update the Docker Architecture**

We need to add a PostgreSQL testing database to our environment and install the necessary database drivers in our lab runner.

1. **Update `docker-compose.yml**`:
Add the `postgres-db` service to your stack.

```yaml
services:
  # ... (Keep ollama-service, mlflow-server, and unsloth-trainer)

  # Service 5: The Evaluation Database
  postgres-db:
    image: postgres:15-alpine
    container_name: sql_eval_db
    environment:
      - POSTGRES_USER=testuser
      - POSTGRES_PASSWORD=testpass
      - POSTGRES_DB=company_db
    ports:
      - "5432:5432"

  # Update lab-runner to be able to talk to the DB
  lab-runner:
    image: python:3.10-slim
    container_name: sql_lab_runner
    volumes:
      - .:/app
    working_dir: /app
    command: tail -f /dev/null
    environment:
      - OLLAMA_HOST=http://ollama-service:11434
      - DB_URI=postgresql://testuser:testpass@postgres-db:5432/company_db
```

2. **Update `requirements.txt**`:
   Add the tools needed for data comparison and database connections.

```text
langchain
langchain-community
langchain-ollama
dvc
pandas
psycopg2-binary
pytest

```

3. **Rebuild and Start**:

```bash
docker compose up -d --build

```

---

### **Step 2: Seed the Testing Database**

Before we can evaluate execution accuracy, the database needs the schema and some mock data so the queries have something to return.

Create `src/seed_db.py`:

```python
import os
import psycopg2

DB_URI = os.getenv("DB_URI", "postgresql://testuser:testpass@postgres-db:5432/company_db")

schema_sql = """
DROP TABLE IF EXISTS company_sales_2024;

CREATE TABLE company_sales_2024 (
    region_id VARCHAR(50),
    total_arr_cents BIGINT,
    sales_rep_uid UUID,
    deal_closed_at TIMESTAMP,
    product_tier VARCHAR(50)
);

INSERT INTO company_sales_2024 (region_id, total_arr_cents, sales_rep_uid, deal_closed_at, product_tier)
VALUES
    ('NA', 5000000, gen_random_uuid(), '2024-01-15', 'Enterprise'),
    ('NA', 1500000, gen_random_uuid(), '2024-02-20', 'SMB'),
    ('EMEA', 3000000, gen_random_uuid(), '2024-03-10', 'Enterprise'),
    ('APAC', 800000, gen_random_uuid(), '2024-04-05', 'Starter');
"""

def seed():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    print("Executing schema and seeding data...")
    cur.execute(schema_sql)
    conn.commit()

    cur.close()
    conn.close()
    print("Database seeded successfully.")

if __name__ == "__main__":
    seed()

```

Run the seeder inside the container:

```bash
docker exec sql_lab_runner python src/seed_db.py

```

---

### **Step 3: The Execution Evaluator**

We will write a test script that takes a small "Holdout" dataset (queries the model did not see during training), generates predictions, runs both queries against Postgres, and compares the resulting Pandas DataFrames.

Create `tests/evaluate_sql.py`:

````python
import os
import json
import pandas as pd
import psycopg2
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP CONNECTIONS
DB_URI = os.getenv("DB_URI")

# Note: We are loading the fine-tuned model we exported in Phase 3
# Ensure you loaded the LoRA adapter into Ollama using a Modelfile first
llm = ChatOllama(
    base_url="http://ollama-service:11434",
    model="qwen-sql-finetuned:latest", # Your local fine-tuned tag
    temperature=0
)

# 2. HOLDOUT TEST DATA (Ground Truth)
# These are questions the model wasn't trained on
test_cases = [
    {
        "question": "What is the total revenue in dollars for the NA region?",
        "ground_truth_sql": "SELECT SUM(total_arr_cents) / 100 AS total_dollars FROM company_sales_2024 WHERE region_id = 'NA';"
    },
    {
        "question": "Count the number of Enterprise deals.",
        "ground_truth_sql": "SELECT COUNT(*) FROM company_sales_2024 WHERE product_tier = 'Enterprise';"
    }
]

SCHEMA = """
Table: company_sales_2024
Columns: region_id, total_arr_cents, sales_rep_uid, deal_closed_at, product_tier
"""

prompt = ChatPromptTemplate.from_template(
    "You are a Postgres SQL expert. Given the schema, write the SQL query. Output ONLY the SQL.\n\nSchema: {schema}\n\nQuestion: {question}"
)
chain = prompt | llm

def execute_query(query: str) -> pd.DataFrame:
    """Executes SQL and returns a Pandas DataFrame."""
    try:
        conn = psycopg2.connect(DB_URI)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty DF on SQL syntax error

def evaluate():
    correct = 0
    total = len(test_cases)

    print(f"Starting Execution Evaluation on {total} test cases...\n")

    for idx, case in enumerate(test_cases):
        print(f"Test {idx + 1}: {case['question']}")

        # A. Generate Prediction
        response = chain.invoke({"schema": SCHEMA, "question": case["question"]})
        predicted_sql = response.content.replace("```sql", "").replace("```", "").strip()

        # B. Execute Both
        df_truth = execute_query(case["ground_truth_sql"])
        df_pred = execute_query(predicted_sql)

        # C. Compare DataFrames
        # We use .equals() which checks if shape and elements are exactly the same
        # For production, you'd add logic to sort the dataframes to ignore row ordering
        if df_truth.empty and df_pred.empty:
            print("❌ Both failed to execute.")
        elif df_truth.equals(df_pred):
            print(f"✅ PASS: Results match.")
            correct += 1
        else:
            print(f"❌ FAIL: Results differ.")
            print(f"   Predicted SQL: {predicted_sql}")
            print(f"   Expected SQL : {case['ground_truth_sql']}")
        print("-" * 40)

    accuracy = (correct / total) * 100
    print(f"\nExecution Accuracy: {accuracy}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()

````

---

### **Phase 4 Verification**

To run your evaluation suite:

```bash
docker exec sql_lab_runner python tests/evaluate_sql.py

```

**Expected Outcome:**
You should see an output detailing which tests passed and the final `Execution Accuracy: 100%`.

If it fails, look closely at the `Predicted SQL` printed in the terminal. Is the model still forgetting to divide `total_arr_cents` by 100? If so, your Phase 3 fine-tuning data might not have had enough examples of that specific rule. This tight feedback loop is exactly what MLOps is designed to support.

**Ready for Phase 5?**
If your model passes the execution tests, it is ready for deployment. Phase 5 involves packaging this adapter into a `.gguf` file via an `Ollama Modelfile` so it can be served continuously as a production API.

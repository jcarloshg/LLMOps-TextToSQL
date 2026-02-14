import os
import json
import pandas as pd
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. SETUP CONNECTIONS
DB_URI = os.getenv("DB_URI")

# Load fine-tuned model directly from lora_model folder
print("Loading fine-tuned model from lora_model/...")
tokenizer = AutoTokenizer.from_pretrained("lora_model")
model = AutoModelForCausalLM.from_pretrained(
    "lora_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. HOLDOUT TEST DATA (Ground Truth)
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

SYSTEM_PROMPT = "You are a Postgres SQL expert. Given the schema, write a single SQL query to answer the question. Return ONLY the SQL query, nothing else."

def generate_sql(question: str) -> str:
    """Generate SQL using the fine-tuned model."""
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{SCHEMA}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0,
            top_p=1.0
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract SQL from response
    sql = response.split("<|im_start|>assistant\n")[-1].strip()

    # Clean up the SQL
    sql = sql.replace("```sql", "").replace("```", "")
    if "SELECT" in sql.upper():
        start_idx = sql.upper().find("SELECT")
        sql = sql[start_idx:]
    if ";" in sql:
        sql = sql[:sql.find(";") + 1]

    return sql.strip()


def execute_query(query: str) -> pd.DataFrame:
    """Executes SQL and returns a Pandas DataFrame."""
    try:
        engine = create_engine(DB_URI)
        df = pd.read_sql_query(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()


def evaluate():
    correct = 0
    total = len(test_cases)

    print(f"Starting Evaluation on {total} test cases with fine-tuned model...\n")

    for idx, case in enumerate(test_cases):
        print(f"Test {idx + 1}: {case['question']}")

        # A. Generate Prediction
        predicted_sql = generate_sql(case["question"])
        print(f"   Generated SQL: {predicted_sql}")

        # B. Execute Both
        df_truth = execute_query(case["ground_truth_sql"])
        df_pred = execute_query(predicted_sql)

        # C. Compare DataFrames
        if df_truth.empty and df_pred.empty:
            print("❌ Both failed to execute.")
        elif df_truth.empty or df_pred.empty:
            print(f"❌ FAIL: One query failed, other succeeded.")
        else:
            # Compare by values only
            truth_values = df_truth.reset_index(drop=True).values
            pred_values = df_pred.reset_index(drop=True).values

            try:
                import numpy as np
                match = np.allclose(truth_values, pred_values, rtol=1e-5)
                if match:
                    print(f"✅ PASS: Results match.")
                    correct += 1
                else:
                    print(f"❌ FAIL: Results differ.")
            except (TypeError, ValueError):
                match = (truth_values == pred_values).all()
                if match:
                    print(f"✅ PASS: Results match.")
                    correct += 1
                else:
                    print(f"❌ FAIL: Results differ.")
        print("-" * 40)

    accuracy = (correct / total) * 100
    print(f"\nEvaluation Accuracy: {accuracy}% ({correct}/{total})")


if __name__ == "__main__":
    evaluate()

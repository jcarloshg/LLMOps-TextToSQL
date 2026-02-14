import os
import json
import pandas as pd
from sqlalchemy import create_engine
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP CONNECTIONS
DB_URI = os.getenv("DB_URI")

# Using base model for evaluation
# TODO: After training completes, create a Modelfile to load the fine-tuned LoRA adapter
llm = ChatOllama(
    base_url="http://ollama-service:11434",
    model="qwen2.5-coder:1.5b",  # Base model (fine-tuned version will be integrated later)
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
    "You are a Postgres SQL expert. Given the schema, write a single SQL query to answer the question. Return ONLY the SQL query, nothing else.\n\nSchema: {schema}\n\nQuestion: {question}\n\nSQL Query:"
)
chain = prompt | llm


def execute_query(query: str) -> pd.DataFrame:
    """Executes SQL and returns a Pandas DataFrame."""
    try:
        engine = create_engine(DB_URI)
        df = pd.read_sql_query(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()  # Return empty DF on SQL syntax error


def evaluate():
    correct = 0
    total = len(test_cases)

    print(f"Starting Execution Evaluation on {total} test cases...\n")

    for idx, case in enumerate(test_cases):
        print(f"Test {idx + 1}: {case['question']}")

        # A. Generate Prediction
        response = chain.invoke(
            {"schema": SCHEMA, "question": case["question"]})

        # Extract SQL from response (handle various formats)
        predicted_sql = response.content
        # Remove markdown code blocks
        predicted_sql = predicted_sql.replace("```sql", "").replace("```", "")
        # Extract first SELECT statement
        if "SELECT" in predicted_sql.upper():
            start_idx = predicted_sql.upper().find("SELECT")
            predicted_sql = predicted_sql[start_idx:]
        # Remove trailing explanations (anything after semicolon)
        if ";" in predicted_sql:
            predicted_sql = predicted_sql[:predicted_sql.find(";") + 1]
        predicted_sql = predicted_sql.strip()

        # B. Execute Both
        df_truth = execute_query(case["ground_truth_sql"])
        df_pred = execute_query(predicted_sql)

        # C. Compare DataFrames
        # Compare by values only, ignoring column names and order
        if df_truth.empty and df_pred.empty:
            print("❌ Both failed to execute.")
        elif df_truth.empty or df_pred.empty:
            print(f"❌ FAIL: One query failed, other succeeded.")
        else:
            # Reset indices and compare values only (ignore column names)
            truth_values = df_truth.reset_index(drop=True).values
            pred_values = df_pred.reset_index(drop=True).values

            # Compare values with tolerance for floats
            import numpy as np
            try:
                match = np.allclose(truth_values, pred_values, rtol=1e-5)
                if match:
                    print(f"✅ PASS: Results match (semantically equivalent).")
                    correct += 1
                else:
                    print(f"❌ FAIL: Results differ.")
            except (TypeError, ValueError):
                # Fallback for non-numeric data
                match = (truth_values == pred_values).all()
                if match:
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

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
4. Output strict JSON format list: [{{"question": "...", "sql": "..."}}, ...]

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
        content = response.content.replace(
            "```json", "").replace("```", "").strip()
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

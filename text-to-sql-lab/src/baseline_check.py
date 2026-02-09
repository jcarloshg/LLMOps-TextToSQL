
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

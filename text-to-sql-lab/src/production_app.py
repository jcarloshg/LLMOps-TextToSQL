import os
import psycopg2
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# 1. SETUP MLOPS TRACING
# This connects LangChain to our local Dockerized Phoenix server
print("Connecting to Arize Phoenix...")
tracer_provider = register(
    project_name="text-to-sql-production",
    endpoint="http://phoenix:6006/v1/traces"
)
# Automatically capture all LangChain calls
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# 2. INITIALIZE PRODUCTION MODEL
# Notice we don't need the system prompt here; we baked it into the Modelfile in Phase 5!
llm = ChatOllama(
    base_url="http://ollama-service:11434",
    model="qwen-sql-prod",
    temperature=0
)
prompt = ChatPromptTemplate.from_template("Question: {question}")
chain = prompt | llm

DB_URI = os.getenv(
    "DB_URI", "postgresql://testuser:testpass@postgres-db:5432/company_db")


def handle_user_request(question: str):
    print(f"\nUser asked: '{question}'")

    # A. Generate SQL (This step is automatically traced by Phoenix)
    response = chain.invoke({"question": question})
    sql_query = response.content.replace(
        "```sql", "").replace("```", "").strip()

    print(f"Generated SQL: {sql_query}")

    # B. Execute against the database
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        cur.execute(sql_query)
        results = cur.fetchall()
        cur.close()
        conn.close()
        print(f"✅ DB Success! Rows returned: {len(results)}")
        return results
    except Exception as e:
        # C. Catch and report syntax errors
        print(f"❌ DB Error: {e}")
        return str(e)


if __name__ == "__main__":
    # Simulate a successful query
    handle_user_request("Show me the total revenue for the SMB tier.")

    # Simulate an edge-case query that might break the model
    # (e.g., asking for a column 'employee_count' that doesn't exist)
    handle_user_request("How many employees are in the NA region?")

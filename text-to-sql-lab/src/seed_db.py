import os
import psycopg2

DB_URI = os.getenv(    "DB_URI", "postgresql://testuser:testpass@postgres-db:5432/company_db")

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

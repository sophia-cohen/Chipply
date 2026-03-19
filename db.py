import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """Read from env vars first, fall back to Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return None


DATABASE_URL = _get_secret("DATABASE_URL")
AFFINITY_DATABASE_URL = _get_secret("AFFINITY_DATABASE_URL")


def get_connection(db="main"):
    url = AFFINITY_DATABASE_URL if db == "affinity" else DATABASE_URL
    return psycopg2.connect(url)


def _introspect_db(conn) -> tuple[list, dict, dict, dict, dict]:
    """Introspect a single database connection and return raw schema info."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Get all tables and their columns
    cur.execute("""
        SELECT
            t.table_schema,
            t.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            c.character_maximum_length
        FROM information_schema.tables t
        JOIN information_schema.columns c
            ON t.table_name = c.table_name AND t.table_schema = c.table_schema
        WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
            AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_schema, t.table_name, c.ordinal_position
    """)
    columns = cur.fetchall()

    # Get primary keys
    cur.execute("""
        SELECT
            tc.table_schema,
            tc.table_name,
            kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
    """)
    pks = {}
    for row in cur.fetchall():
        key = (row["table_schema"], row["table_name"])
        pks.setdefault(key, []).append(row["column_name"])

    # Get foreign keys
    cur.execute("""
        SELECT
            tc.table_schema,
            tc.table_name,
            kcu.column_name,
            ccu.table_schema AS ref_schema,
            ccu.table_name AS ref_table,
            ccu.column_name AS ref_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
    """)
    fks = {}
    for row in cur.fetchall():
        key = (row["table_schema"], row["table_name"])
        fks.setdefault(key, []).append(row)

    # Get row counts per table
    tables_seen = set()
    for col in columns:
        tables_seen.add((col["table_schema"], col["table_name"]))

    row_counts = {}
    for schema, table in tables_seen:
        try:
            cur.execute(
                f'SELECT COUNT(*) as cnt FROM "{schema}"."{table}"'
            )
            row_counts[(schema, table)] = cur.fetchone()["cnt"]
        except Exception:
            conn.rollback()
            row_counts[(schema, table)] = "unknown"

    # Get sample values for each table (first 3 rows)
    sample_data = {}
    for schema, table in tables_seen:
        try:
            cur.execute(
                f'SELECT * FROM "{schema}"."{table}" LIMIT 3'
            )
            rows = cur.fetchall()
            if rows:
                sample_data[(schema, table)] = rows
        except Exception:
            conn.rollback()

    cur.close()
    conn.close()

    return columns, pks, fks, row_counts, sample_data


def _format_schema(label: str, columns, pks, fks, row_counts, sample_data) -> list[str]:
    """Format introspected schema info into readable lines."""
    lines = [f"\n# {label}\n"]
    current_table = None

    for col in columns:
        key = (col["table_schema"], col["table_name"])
        table_label = f'{col["table_schema"]}.{col["table_name"]}'

        if key != current_table:
            current_table = key
            count = row_counts.get(key, "?")
            lines.append(f"\n## Table: {table_label} ({count} rows)")

            pk_cols = pks.get(key, [])
            if pk_cols:
                lines.append(f"  Primary key: {', '.join(pk_cols)}")

            table_fks = fks.get(key, [])
            for fk in table_fks:
                lines.append(
                    f"  FK: {fk['column_name']} -> "
                    f"{fk['ref_schema']}.{fk['ref_table']}.{fk['ref_column']}"
                )

            lines.append("  Columns:")

        nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
        dtype = col["data_type"]
        if col["character_maximum_length"]:
            dtype += f"({col['character_maximum_length']})"
        default = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
        lines.append(f"    - {col['column_name']}: {dtype} {nullable}{default}")

    # Add sample data
    for (schema, table), rows in sample_data.items():
        lines.append(f"\n### Sample data: {schema}.{table}")
        if rows:
            cols = list(rows[0].keys())
            lines.append("  " + " | ".join(cols))
            lines.append("  " + " | ".join(["---"] * len(cols)))
            for row in rows:
                vals = [str(v)[:50] if v is not None else "NULL" for v in row.values()]
                lines.append("  " + " | ".join(vals))

    return lines


def get_schema() -> str:
    """Introspect both databases and return a combined schema description."""
    lines = []

    conn_main = get_connection("main")
    main_info = _introspect_db(conn_main)
    lines.extend(_format_schema("Main Database", *main_info))

    if AFFINITY_DATABASE_URL:
        conn_affinity = get_connection("affinity")
        affinity_info = _introspect_db(conn_affinity)
        lines.extend(_format_schema("Affinity Interactions Database", *affinity_info))
        lines.append("\n# Cross-Database Relationships")
        lines.append("The Main Database contains an 'Affinity' table that maps entities to their Affinity IDs.")
        lines.append("Use these IDs to join with tables in the Affinity Interactions Database.")
        lines.append("To query across both databases, use run_sql with db='main' or db='affinity' as needed.")

    return "\n".join(lines)


def run_query(sql: str, db: str = "main") -> tuple[list[dict], list[str]]:
    """Execute a read-only SQL query and return (rows, column_names)."""
    conn = get_connection(db)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        return [dict(r) for r in rows], col_names
    finally:
        cur.close()
        conn.close()

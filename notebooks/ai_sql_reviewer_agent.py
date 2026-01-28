# Databricks notebook source
# DBTITLE 1,Untitled
print("Notebook is running")

# COMMAND ----------

# DBTITLE 1,Untitled
spark.sql("SELECT 1 AS test_column").display()

# COMMAND ----------

# DBTITLE 1,Cell 3
spark.sql("SHOW CATALOGS").display()

# COMMAND ----------

# DBTITLE 1,Cell 4
spark.sql("USE CATALOG winfo_dbx")
print("Using winfo_dbx")

# COMMAND ----------

spark.sql("""
CREATE SCHEMA IF NOT EXISTS sql_agent
""")

spark.sql("USE SCHEMA sql_agent")

print("Schema sql_agent ready")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE customers AS
SELECT
  id AS customer_id,
  concat('customer_', id) AS name,
  CASE 
    WHEN id % 10 = 0 THEN NULL 
    ELSE concat('user', id, '@example.com') 
  END AS email,
  date_sub(current_date(), CAST(id % 365 AS INT)) AS signup_date
FROM range(1, 50001)
""")

print("customers table created")

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TABLE orders AS
SELECT
  id AS order_id,
  (id % 50000) + 1 AS customer_id,
  current_timestamp() - INTERVAL 720 HOURS AS order_ts,
  CAST((id % 10000) / 10.0 AS DOUBLE) AS amount,
  CASE 
    WHEN id % 7 = 0 THEN 'refunded' 
    ELSE 'completed' 
  END AS status
FROM range(1, 300001)
""")

print("orders table created")

# COMMAND ----------

spark.sql("ANALYZE TABLE customers COMPUTE STATISTICS")
spark.sql("ANALYZE TABLE orders COMPUTE STATISTICS")

print("Statistics computed")

# COMMAND ----------

spark.sql("SELECT COUNT(*) AS customers FROM customers").display()
spark.sql("SELECT COUNT(*) AS orders FROM orders").display()

# COMMAND ----------

spark.sql("USE CATALOG winfo_dbx")
spark.sql("USE SCHEMA sql_agent")
print("Using winfo_dbx.sql_agent")

# COMMAND ----------

spark.sql("SHOW TABLES").display()

# COMMAND ----------

BAD_QUERY_1 = """
SELECT *
FROM orders o
JOIN customers c
  ON o.customer_id = c.customer_id
WHERE c.email LIKE '%@example.com%'
  AND o.status = 'completed'
ORDER BY o.order_ts DESC
"""

BAD_QUERY_2 = """
SELECT c.customer_id, c.name, SUM(o.amount) AS total_spend
FROM customers c
LEFT JOIN orders o
  ON c.customer_id = o.customer_id
WHERE date(o.order_ts) >= date_sub(current_date(), 30)
GROUP BY c.customer_id, c.name
"""

print("Bad queries loaded")

# COMMAND ----------

from typing import Dict

def tool_explain(sql_text: str, mode: str = "FORMATTED") -> Dict[str, str]:
    """
    MCP-like tool: returns explain plan text for a given SQL query.
    mode: "FORMATTED" is more readable, "EXTENDED" includes more details.
    """
    mode = mode.upper().strip()
    if mode not in {"FORMATTED", "EXTENDED", "SIMPLE"}:
        raise ValueError("mode must be one of: FORMATTED, EXTENDED, SIMPLE")

    explain_sql = f"EXPLAIN {mode} {sql_text}"
    rows = spark.sql(explain_sql).collect()

    # Databricks/Spark returns explain output as rows of strings; join them into one blob.
    plan_text = "\n".join(r[0] for r in rows if r and r[0] is not None)

    return {
        "mode": mode,
        "explain_sql": explain_sql,
        "plan_text": plan_text,
    }

# COMMAND ----------

result = tool_explain(BAD_QUERY_1, mode="FORMATTED")
print(result["plan_text"][:2000])  # print first ~2000 chars

# COMMAND ----------

# STEP 4.1 — Detect Model Serving / Foundation Model access via Databricks SDK (best signal)

def check_model_serving():
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        eps = list(w.serving_endpoints.list())
        print(f"✅ Model Serving API reachable. Endpoints visible: {len(eps)}")

        # Print up to 10 endpoint names
        for ep in eps[:10]:
            print(" -", getattr(ep, "name", str(ep)))

        return True, eps
    except Exception as e:
        print("❌ Could not access Model Serving via SDK.")
        print("Error:", repr(e))
        return False, None

ok, endpoints = check_model_serving()

# COMMAND ----------

LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
print("Using LLM endpoint:", LLM_ENDPOINT)

# COMMAND ----------

# DBTITLE 1,Cell 17
import json
from typing import Any, Dict
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def call_llm(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls a Databricks Model Serving endpoint using the Databricks SDK.
    Payload format depends on the endpoint; most chat/instruct models accept a 'messages' structure.
    """
    resp = w.serving_endpoints.query(name=endpoint, inputs=payload)
    # resp is a databricks.sdk service object; convert to dict safely
    return json.loads(resp.as_json())

# COMMAND ----------

# DBTITLE 1,Cell 18
RESPONSE_KEYS = {
    "summary",
    "risks",
    "performance_findings",
    "cost_signals",
    "correctness_findings",
    "recommended_rewrites",
    "optimized_sql",
    "why_this_is_better",
    "next_checks",
    "confidence"
}

def validate_review_json(obj: Dict[str, Any]) -> None:
    missing = RESPONSE_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Missing keys in model output: {sorted(missing)}")

    if not isinstance(obj["risks"], list):
        raise ValueError("risks must be a list")
    if not isinstance(obj["recommended_rewrites"], list):
        raise ValueError("recommended_rewrites must be a list")
    if not (isinstance(obj["confidence"], (int, float)) and 0 <= obj["confidence"]):
        raise ValueError("confidence must be a non-negative number")

# COMMAND ----------

# DBTITLE 1,Cell 19
import json
from typing import Any, Dict, Optional
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

PRIMARY_ENDPOINT = "databricks-claude-sonnet-4-5"
FALLBACK_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"

# Update the required keys to include semantics_risk in each rewrite object (we validate loosely)
RESPONSE_KEYS = {
    "summary",
    "risks",
    "performance_findings",
    "cost_signals",
    "correctness_findings",
    "recommended_rewrites",
    "optimized_sql",
    "why_this_is_better",
    "next_checks",
    "confidence"
}

def validate_review_json(obj: Dict[str, Any]) -> None:
    missing = RESPONSE_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Missing keys in model output: {sorted(missing)}")

    if not isinstance(obj["risks"], list):
        raise ValueError("risks must be a list")
    if not isinstance(obj["recommended_rewrites"], list):
        raise ValueError("recommended_rewrites must be a list")
    if not (isinstance(obj["confidence"], (int, float)) and 0 <= obj["confidence"] <= 1):
        raise ValueError("confidence must be a number between 0 and 1")

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _extract_text(raw: Dict[str, Any]) -> str:
    text = None
    if "choices" in raw and raw["choices"]:
        text = raw["choices"][0].get("message", {}).get("content") or raw["choices"][0].get("text")
    elif "predictions" in raw and raw["predictions"]:
        pred0 = raw["predictions"][0]
        text = pred0.get("content") or pred0.get("text") or pred0.get("result")
    elif "output" in raw:
        text = raw["output"]

    if not text or not isinstance(text, str):
        raise RuntimeError(f"Could not extract text from model response. Raw keys: {list(raw.keys())}")

    return _strip_markdown_fences(text)

def tool_llm_review(
    sql_query: str,
    plan_text: str,
    *,
    endpoint: Optional[str] = None,
    max_tokens: int = 2500
) -> Dict[str, Any]:
    """
    MCP-like tool: uses LLM to review and propose optimizations based on SQL + EXPLAIN plan text.
    Hardened:
      - requires semantics_risk field per rewrite
      - chooses optimized_sql preferring semantics_risk=false
      - fallback endpoint
    """
    ep_primary = endpoint or PRIMARY_ENDPOINT

    system = (
        "You are a senior Databricks performance engineer. "
        "You MUST preserve query semantics. "
        "If a proposed rewrite might change semantics, you must flag it in correctness_findings "
        "AND set semantics_risk=true on that rewrite. "
        "Return STRICT JSON only. No markdown."
    )

    user = f"""
SQL_QUERY:
{sql_query}

EXPLAIN_PLAN_TEXT:
{plan_text}

Return JSON with EXACT keys:
- summary (string)
- risks (array of objects: {{type, detail}})
- performance_findings (array of strings)
- cost_signals (array of strings)
- correctness_findings (array of strings)
- recommended_rewrites (array of objects: {{
    title,
    rewrite_sql,
    notes,
    semantics_risk
  }})
- optimized_sql (string)
- why_this_is_better (array of strings)
- next_checks (array of strings)
- confidence (number 0..1)

Hard rules:
- Do NOT change filters (LIKE patterns, date ranges), join type, or grouping unless semantics_risk=true.
- LIMIT is semantics-changing unless the user explicitly asked for top-N; if you add LIMIT, set semantics_risk=true.
- Prefer explicit column selection instead of SELECT *.
- Prefer range predicates over wrapping timestamp columns in functions.
- optimized_sql should be the best rewrite with semantics_risk=false if possible; otherwise keep original SQL.
""".strip()

    payload = {
        "messages": [
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system),
            ChatMessage(role=ChatMessageRole.USER, content=user),
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    last_err = None
    for ep in [ep_primary, FALLBACK_ENDPOINT]:
        try:
            raw = call_llm(ep, payload)
            text = _extract_text(raw)
            obj = json.loads(text)
            validate_review_json(obj)

            # Safety: if optimized_sql is empty, default to original
            if not obj.get("optimized_sql"):
                obj["optimized_sql"] = sql_query

            return obj
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Both primary and fallback endpoints failed. Last error: {repr(last_err)}")


# COMMAND ----------

plan = tool_explain(BAD_QUERY_1, mode="FORMATTED")["plan_text"]
review = tool_llm_review(BAD_QUERY_1, plan)

print("Summary:", review["summary"])
print("Confidence:", review["confidence"])
print("Top risks:", review["risks"][:2])
print("\nOptimized SQL (first 500 chars):\n", review["optimized_sql"][:500])


# COMMAND ----------

# Clear any old widgets (safe)
try:
    dbutils.widgets.removeAll()
except Exception:
    pass

dbutils.widgets.text("sql_query", BAD_QUERY_1.strip(), "SQL Query (edit me)")
dbutils.widgets.dropdown("explain_mode", "FORMATTED", ["FORMATTED", "EXTENDED", "SIMPLE"], "EXPLAIN Mode")
dbutils.widgets.text("llm_endpoint", LLM_ENDPOINT, "LLM Endpoint")

print("Widgets created. Edit the SQL in the widget UI above, then run the next cell.")


# COMMAND ----------

def print_review(review: dict, *, max_items: int = 6) -> None:
    print("\n" + "="*80)
    print("AI SQL REVIEW")
    print("="*80)
    print("Summary:", review.get("summary", ""))
    print("Confidence:", review.get("confidence", ""))

    print("\n--- Risks ---")
    for r in (review.get("risks") or [])[:max_items]:
        print(f"- [{r.get('type','')}] {r.get('detail','')}")

    print("\n--- Performance findings ---")
    for s in (review.get("performance_findings") or [])[:max_items]:
        print(f"- {s}")

    print("\n--- Cost signals ---")
    for s in (review.get("cost_signals") or [])[:max_items]:
        print(f"- {s}")

    print("\n--- Correctness findings ---")
    for s in (review.get("correctness_findings") or [])[:max_items]:
        print(f"- {s}")

    print("\n--- Recommended rewrites (titles) ---")
    for rw in (review.get("recommended_rewrites") or [])[:max_items]:
        print(f"- {rw.get('title','(no title)')}")

    print("\n--- Optimized SQL ---")
    print(review.get("optimized_sql", "").strip())

    print("\n--- Why this is better ---")
    for s in (review.get("why_this_is_better") or [])[:max_items]:
        print(f"- {s}")

    print("\n--- Next checks ---")
    for s in (review.get("next_checks") or [])[:max_items]:
        print(f"- {s}")


# COMMAND ----------

sql_query = dbutils.widgets.get("sql_query")
explain_mode = dbutils.widgets.get("explain_mode")
endpoint = dbutils.widgets.get("llm_endpoint")

print("Running review with:")
print(" - explain_mode:", explain_mode)
print(" - endpoint:", endpoint)

plan = tool_explain(sql_query, mode=explain_mode)["plan_text"]

print("\n" + "="*80)
print("EXPLAIN PLAN (first 2000 chars)")
print("="*80)
print(plan[:2000])

review = tool_llm_review(sql_query, plan, endpoint=endpoint)
print_review(review)

opt_sql = review.get("optimized_sql", "")
risky, issues = detect_semantic_risks(sql_query, opt_sql)

print("\n" + "="*80)
print("GUARDRAIL CHECK (optimized_sql)")
print("="*80)
if risky:
    print("⚠️ Potential semantic changes detected:")
    for i in issues:
        print(" -", i)
else:
    print("✅ No obvious semantic changes detected by heuristics.")

safe_sql = pick_safest_sql(sql_query, review)

print("\n" + "="*80)
print("SAFEST SQL TO USE")
print("="*80)
print(safe_sql)


# COMMAND ----------

PRIMARY_ENDPOINT = "databricks-claude-sonnet-4-5"
FALLBACK_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"

print("Primary:", PRIMARY_ENDPOINT)
print("Fallback:", FALLBACK_ENDPOINT)


# COMMAND ----------

plan = tool_explain(BAD_QUERY_1, mode="FORMATTED")["plan_text"]
review = tool_llm_review(BAD_QUERY_1, plan)

print("Optimized SQL:\n", review["optimized_sql"])
print("\nFirst rewrite object keys:", list(review["recommended_rewrites"][0].keys()) if review["recommended_rewrites"] else "no rewrites")


# COMMAND ----------

import re
from typing import Tuple, List

def _normalize_sql(s: str) -> str:
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def detect_semantic_risks(original_sql: str, rewritten_sql: str) -> Tuple[bool, List[str]]:
    o = _normalize_sql(original_sql)
    r = _normalize_sql(rewritten_sql)

    issues = []

    # JOIN type changes
    join_markers = [" left join ", " right join ", " full join ", " inner join ", " cross join "]
    for jm in join_markers:
        if (jm in o) != (jm in r):
            issues.append(f"Join type presence changed for: {jm.strip()}")

    # LIMIT changes
    if (" limit " in o) != (" limit " in r):
        issues.append("LIMIT presence changed (can change result set size).")

    # LIKE pattern changes (simple heuristic)
    o_likes = re.findall(r"like\s+'([^']+)'", o)
    r_likes = re.findall(r"like\s+'([^']+)'", r)
    if o_likes != r_likes:
        issues.append(f"LIKE patterns changed: original={o_likes} rewritten={r_likes}")

    # GROUP BY changes
    if (" group by " in o) != (" group by " in r):
        issues.append("GROUP BY presence changed (can change aggregation semantics).")

    risky = len(issues) > 0
    return risky, issues


# COMMAND ----------

from typing import Dict, Any

def pick_safest_sql(original_sql: str, review: Dict[str, Any]) -> str:
    rewrites = review.get("recommended_rewrites") or []
    for rw in rewrites:
        if rw.get("semantics_risk") is False and rw.get("rewrite_sql"):
            return rw["rewrite_sql"]
    # fallback: optimized_sql if it exists and seems safe by heuristic
    opt = review.get("optimized_sql") or ""
    if opt:
        risky, _ = detect_semantic_risks(original_sql, opt)
        if not risky:
            return opt
    return original_sql


# COMMAND ----------

safe_sql = pick_safest_sql(sql_query, review)

plan_before = tool_explain(sql_query, mode="FORMATTED")["plan_text"]
plan_after  = tool_explain(safe_sql,  mode="FORMATTED")["plan_text"]

print("\n" + "="*80)
print("PLAN BEFORE (first 1200 chars)")
print("="*80)
print(plan_before[:1200])

print("\n" + "="*80)
print("PLAN AFTER (first 1200 chars)")
print("="*80)
print(plan_after[:1200])

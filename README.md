AI SQL Reviewer & Optimizer Agent (Databricks)

Goal
A notebook-based agent that reviews Spark SQL queries for performance, cost, and correctness risks, grounded in the actual Photon/Spark execution plan, and proposes safe optimizations.

How it works

User provides a SQL query in the notebook widget

Tool layer extracts the execution plan via EXPLAIN

An LLM (Databricks Model Serving) produces a structured JSON review:

risks and findings

rewrite candidates

an “optimized” proposal

Guardrails detect potential semantic changes (e.g., JOIN/LIMIT/LIKE pattern changes)

The notebook selects and displays the safest rewrite automatically

Tech

Databricks Notebooks (Python + Spark SQL)

Databricks Model Serving endpoint: databricks-claude-sonnet-4-5

Fallback endpoint: databricks-meta-llama-3-1-8b-instruct

MCP-style tools (in-notebook)

tool_explain(sql, mode) → returns EXPLAIN plan text

tool_llm_review(sql, plan) → returns strict JSON review + rewrite candidates (semantics_risk)

detect_semantic_risks(original, rewrite) → heuristic guardrail for silent semantic changes

pick_safest_sql(original, review) → selects safest rewrite candidate for execution/diffing

Safety & Governance

The agent labels rewrites with semantics_risk and warns when changes could alter results.

Safe rewrite selection prefers non-risky optimizations (projection reduction, predicate improvements, etc.).

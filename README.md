# AI SQL Reviewer & Optimizer Agent (Databricks)

## Goal
A notebook-based agent that reviews Spark SQL queries for **performance**, **cost**, and **correctness** risks, grounded in the actual **Photon / Spark execution plan**, and proposes **safe optimizations**.

---

## How It Works
1. The user provides a SQL query via a Databricks notebook widget  
2. A tool layer extracts the execution plan using `EXPLAIN`  
3. An LLM (Databricks Model Serving) produces a **structured JSON review**, including:
   - performance and cost risks
   - correctness considerations
   - rewrite candidates
   - an optimized proposal
4. Guardrails detect potential **semantic changes** (e.g. JOIN, LIMIT, LIKE pattern changes)
5. The notebook automatically selects and displays the **safest rewrite**

---

## Tech Stack
- Databricks Notebooks (Python + Spark SQL)
- Databricks Model Serving
  - Primary endpoint: `databricks-claude-sonnet-4-5`
  - Fallback endpoint: `databricks-meta-llama-3-1-8b-instruct`

---

## MCP-Style Tools (In-Notebook)
- `tool_explain(sql, mode)`  
  Returns Spark / Photon `EXPLAIN` plan text

- `tool_llm_review(sql, plan)`  
  Produces a strict JSON review with rewrite candidates and `semantics_risk`

- `detect_semantic_risks(original, rewrite)`  
  Heuristic guardrail to detect silent semantic changes

- `pick_safest_sql(original, review)`  
  Selects the safest rewrite candidate for execution or diffing

---

## Safety & Governance
- Rewrite candidates are explicitly labeled with `semantics_risk`
- Guardrails warn when optimizations may change query meaning
- Safe selection prefers **non-risky optimizations** such as:
  - column projection reduction
  - predicate pushdown improvements
  - avoiding unnecessary shuffles or sorts

---

## Demo Flow
1. Paste a SQL query into the notebook widget  
2. Run the review cell  
3. Inspect the execution plan and AI findings  
4. Review guardrail warnings (if any)  
5. Compare original vs safest optimized SQL

---

## Notes
This project is intended as a **Databricks-native AI agent demo** and focuses on correctness-first optimization rather than aggressive query rewriting.

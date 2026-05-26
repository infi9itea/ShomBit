#!/usr/bin/env python3
"""
eval_all.py — End-to-end evaluation for the EWU RAG pipeline.

Required env vars (same as app.py):
  HF_TOKEN, GROQ_API_KEY   (read by config.py)
  DATA_DIR                 path to KB markdown files

Optional env vars:
  OUT_GOLD        output JSONL  (default: /kaggle/working/gold.jsonl)
  OUT_RESULTS     metrics CSV   (default: /kaggle/working/results_full.csv)
  ABLATION_PREFIX CSV prefix    (default: /kaggle/working/ablation)
"""
import os

DATA_DIR        = os.environ.get("DATA_DIR",        "./data")
OUT_GOLD        = os.environ.get("OUT_GOLD",        "/kaggle/working/gold.jsonl")
OUT_RESULTS     = os.environ.get("OUT_RESULTS",     "/kaggle/working/results_full.csv")
ABLATION_PREFIX = os.environ.get("ABLATION_PREFIX", "/kaggle/working/ablation")

from eval_queries import QUERIES
from batch_eval import load_pipeline, run_queries_from_list
from evaluation import score_gold_file, run_ablation_study, wilcoxon_compare

# 1 — Build pipeline
chain = load_pipeline(DATA_DIR)

# 2 — Run all 60 queries
run_queries_from_list(chain, QUERIES, OUT_GOLD)

# 3 — Retrieval + generation metrics + RAGAS
score_gold_file(OUT_GOLD, OUT_RESULTS)

# 4 — Ablation study (full / no_expansion / no_rerank / dense_only)
run_ablation_study(OUT_GOLD, out_prefix=ABLATION_PREFIX)

# 5 — Wilcoxon signed-rank test
for other in ["no_expansion", "no_rerank", "dense_only"]:
    wilcoxon_compare(
        f"{ABLATION_PREFIX}_full.csv",
        f"{ABLATION_PREFIX}_{other}.csv",
        label_a="full",
        label_b=other,
    )

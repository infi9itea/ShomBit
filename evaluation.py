from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from typing import Dict, List


# ──────────────────────────────────────────────────────────────────
# Retrieval metrics (rank-aware, on the source-URL granularity)
# ──────────────────────────────────────────────────────────────────
def hit_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def mrr(retrieved: List[str], relevant: set) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], start=1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# ──────────────────────────────────────────────────────────────────
# Lightweight generation metrics (offline fallback)
# ──────────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9]+|[\u0980-\u09FF]+")


def _tok(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def token_f1(pred: str, ref: str) -> float:
    p, r = _tok(pred), _tok(ref)
    if not p or not r:
        return 0.0
    common = 0
    rcopy = list(r)
    for t in p:
        if t in rcopy:
            rcopy.remove(t)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    p, r = _tok(pred), _tok(ref)
    if not p or not r:
        return 0.0
    # LCS length via DP
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# ──────────────────────────────────────────────────────────────────
# Optional RAGAS (LLM-judged faithfulness / relevancy / context metrics)
# ──────────────────────────────────────────────────────────────────
def run_ragas(records: List[dict]) -> Dict[str, float] | None:
    """records: [{question, answer, contexts, reference}]. Needs a judge LLM
    configured (e.g. OPENAI_API_KEY) and `pip install ragas datasets`."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ragas] not available ({e}); skipping LLM-judged metrics.")
        return None

    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r.get("reference", ""),
        }
        for r in records
    ])
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(ds, metrics=metrics)
    return {k: float(v) for k, v in result.items()}


# ──────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────
KS = (1, 3, 5)


def evaluate_system(gold_path: str, out_csv: str, with_ragas: bool = True):
    # Imported lazily so --make-template works without GPU/model deps.
    from huggingface_hub import login
    from config import HF_TOKEN, DATA_DIR
    if HF_TOKEN:
        login(HF_TOKEN)
    from data_pipeline import load_markdown_docs, chunk_documents, scrape_dynamic_docs
    from rag_core import build_vectorstore, Reranker, load_llm, build_rag_chain

    docs = chunk_documents(load_markdown_docs(DATA_DIR)) + chunk_documents(scrape_dynamic_docs())
    vectorstore, bm25, docs = build_vectorstore(docs)
    chain = build_rag_chain(load_llm(HF_TOKEN), vectorstore, bm25, docs, Reranker())

    gold = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]

    rows = []
    ragas_records = []
    for ex in gold:
        res = chain(ex["question"])
        retrieved = res["sources"]
        relevant = set(ex.get("relevant_sources", []))
        row = {
            "question": ex["question"],
            "language": ex.get("language", "unknown"),
            "answer": res["answer"],
            "mrr": mrr(retrieved, relevant),
        }
        for k in KS:
            row[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        if ex.get("reference_answer"):
            row["token_f1"] = token_f1(res["answer"], ex["reference_answer"])
            row["rouge_l"] = rouge_l(res["answer"], ex["reference_answer"])
        rows.append(row)
        ragas_records.append({
            "question": ex["question"], "answer": res["answer"],
            "contexts": res["contexts"], "reference": ex.get("reference_answer", ""),
        })

    _write_csv(rows, out_csv)
    _print_summary(rows)

    if with_ragas:
        scores = run_ragas(ragas_records)
        if scores:
            print("\n=== RAGAS (LLM-judged) ===")
            for k, v in scores.items():
                print(f"  {k:<20} {v:.3f}")


def _write_csv(rows: List[dict], path: str):
    import csv
    if not rows:
        return
    keys = list({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-question results written to {path}")


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _print_summary(rows: List[dict]):
    metric_keys = [k for k in rows[0] if k not in ("question", "language", "answer")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["language"]].append(r)
        groups["ALL"].append(r)

    print("\n=== Retrieval / generation summary (means) ===")
    header = "language".ljust(10) + "n".rjust(4) + "".join(m.rjust(13) for m in metric_keys)
    print(header)
    for lang in sorted(groups):
        rs = groups[lang]
        line = lang.ljust(10) + str(len(rs)).rjust(4)
        for m in metric_keys:
            line += f"{_mean([r.get(m, float('nan')) for r in rs]):13.3f}"
        print(line)


def make_template(path: str):
    samples = [
        {"question": "What is the tuition fee for CSE?", "language": "english",
         "relevant_sources": ["https://ewubd.edu/undergraduate-tuition-fees"], "reference_answer": ""},
        {"question": "CSE vorti deadline kobe?", "language": "banglish",
         "relevant_sources": ["https://ewubd.edu/undergraduate-dates-deadline"], "reference_answer": ""},
        {"question": "কম্পিউটার বিজ্ঞান বিভাগে ভর্তির যোগ্যতা কি?", "language": "bangla",
         "relevant_sources": ["https://ewubd.edu/undergraduate-dates-deadline"], "reference_answer": ""},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote starter gold set to {path}. Aim for >=150 questions, "
          f"balanced across english / bangla / banglish, before reporting numbers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", metavar="PATH")
    ap.add_argument("--gold", metavar="PATH")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--no-ragas", action="store_true")
    args = ap.parse_args()

    if args.make_template:
        make_template(args.make_template)
    elif args.gold:
        evaluate_system(args.gold, args.out, with_ragas=not args.no_ragas)
    else:
        ap.error("pass --make-template PATH or --gold PATH")
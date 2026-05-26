from __future__ import annotations

import argparse
import json
import math
import re
import time
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
    """records: [{question, answer, contexts, reference}].
    Uses Groq (llama-3.3-70b) as the judge LLM so no OpenAI key is needed.
    Requires: pip install ragas datasets langchain-groq langchain-huggingface
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import GROQ_API_KEY, EMBED_MODEL, EMBED_DEVICE
    except Exception as e:
        print(f"[ragas] not available ({e}); skipping LLM-judged metrics.")
        return None

    try:
        judge_llm = LangchainLLMWrapper(
            ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        )
        judge_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": EMBED_DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
        )
        for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
            metric.llm = judge_llm
        answer_relevancy.embeddings = judge_emb
    except Exception as e:
        print(f"[ragas] failed to configure Groq judge ({e}); skipping.")
        return None

    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": (r["contexts"] if isinstance(r["contexts"], list)
                         else [r["contexts"]]),
            "ground_truth": r.get("reference", r.get("ground_truth", "")),
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
    chain = build_rag_chain(load_llm(), vectorstore, bm25, docs, Reranker())

    gold = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]

    rows = []
    ragas_records = []
    for ex in gold:
        t0 = time.perf_counter()
        res = chain(ex["question"])
        latency_s = time.perf_counter() - t0
        retrieved = res["sources"]
        relevant = set(ex.get("relevant_sources", []))
        row = {
            "question": ex["question"],
            "language": ex.get("language", "unknown"),
            "latency_s": round(latency_s, 3),
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


# ──────────────────────────────────────────────────────────────────
# Ablation study
# ──────────────────────────────────────────────────────────────────
_ABLATION_MODES = ["full", "no_expansion", "no_rerank", "dense_only"]
_COMPARE_METRICS = ["hit@1", "hit@3", "mrr", "ndcg@3", "token_f1", "rouge_l", "latency_s"]


def run_ablation_study(gold_path: str, out_prefix: str = "ablation"):
    """Build shared pipeline components once, then run all 4 ablation modes.

    Writes one CSV per mode and prints a side-by-side comparison table.
    """
    import time as _time
    from huggingface_hub import login
    from config import HF_TOKEN, DATA_DIR
    from data_pipeline import load_markdown_docs, chunk_documents, scrape_dynamic_docs
    from rag_core import build_vectorstore, Reranker, load_llm, build_rag_chain

    if HF_TOKEN:
        login(HF_TOKEN)

    print("Loading documents…")
    docs = chunk_documents(load_markdown_docs(DATA_DIR)) + chunk_documents(scrape_dynamic_docs())
    vectorstore, bm25, docs = build_vectorstore(docs)

    print("Loading reranker and LLM…")
    reranker = Reranker()
    llm = load_llm()

    gold = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(gold)} gold queries.\n")

    mode_rows: Dict[str, List[dict]] = {}

    for mode in _ABLATION_MODES:
        print(f"--- ablation: {mode} ---")
        chain = build_rag_chain(llm, vectorstore, bm25, docs, reranker, ablation_mode=mode)
        rows: List[dict] = []
        for i, ex in enumerate(gold, 1):
            t0 = _time.perf_counter()
            res = chain(ex["question"])
            latency_s = _time.perf_counter() - t0
            retrieved = res["sources"]
            relevant = set(ex.get("relevant_sources", []))
            row: dict = {
                "question":  ex["question"],
                "language":  ex.get("language", "unknown"),
                "latency_s": round(latency_s, 3),
                "answer":    res["answer"],
                "mrr":       mrr(retrieved, relevant),
            }
            for k in KS:
                row[f"hit@{k}"]       = hit_at_k(retrieved, relevant, k)
                row[f"recall@{k}"]    = recall_at_k(retrieved, relevant, k)
                row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
                row[f"ndcg@{k}"]      = ndcg_at_k(retrieved, relevant, k)
            gt = ex.get("ground_truth") or ex.get("reference_answer", "")
            if gt:
                row["token_f1"] = token_f1(res["answer"], gt)
                row["rouge_l"]  = rouge_l(res["answer"], gt)
            rows.append(row)
            print(f"  [{i}/{len(gold)}] latency={latency_s:.2f}s")
            if i < len(gold):
                _time.sleep(2.0)   # stay within Groq rate limit

        mode_rows[mode] = rows
        _write_csv(rows, f"{out_prefix}_{mode}.csv")

    # Side-by-side comparison table
    print("\n=== Ablation comparison (macro-average over all queries) ===")
    print(f"{'mode':<16}" + "".join(f"{m:>13}" for m in _COMPARE_METRICS))
    for mode in _ABLATION_MODES:
        rs = mode_rows[mode]
        line = mode.ljust(16)
        for m in _COMPARE_METRICS:
            line += f"{_mean([r.get(m, float('nan')) for r in rs]):>13.3f}"
        print(line)
    print(f"\nPer-mode CSVs: {out_prefix}_<mode>.csv")
    print("Run wilcoxon_compare() to test significance between any two modes.")


# ──────────────────────────────────────────────────────────────────
# Statistical significance (Wilcoxon signed-rank)
# ──────────────────────────────────────────────────────────────────
def wilcoxon_compare(
    csv_a: str,
    csv_b: str,
    label_a: str = "A",
    label_b: str = "B",
    metrics: List[str] | None = None,
):
    """Wilcoxon signed-rank test between two per-question result CSVs.

    Both files must cover the same queries in the same row order (as produced
    by run_ablation_study or evaluate_system). Prints a table of p-values;
    *** p<0.001  ** p<0.01  * p<0.05  ns = not significant.
    """
    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError:
        print("scipy not installed — run: pip install scipy>=1.12.0")
        return

    import csv

    def _load_csv(path: str) -> List[dict]:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows_a = _load_csv(csv_a)
    rows_b = _load_csv(csv_b)
    n = min(len(rows_a), len(rows_b))
    if len(rows_a) != len(rows_b):
        print(f"Warning: row counts differ ({len(rows_a)} vs {len(rows_b)}); "
              f"using first {n} rows.")

    if metrics is None:
        skip = {"question", "language", "answer"}
        metrics = [k for k in rows_a[0] if k not in skip]

    print(f"\n=== Wilcoxon signed-rank: {label_a}  vs  {label_b}  (n={n}) ===")
    print(f"  {'metric':<20} {'mean_'+label_a:>10} {'mean_'+label_b:>10} {'p-value':>10}  sig")
    for m in metrics:
        try:
            xa = [float(rows_a[i].get(m, "nan")) for i in range(n)]
            xb = [float(rows_b[i].get(m, "nan")) for i in range(n)]
            pairs = [(a, b) for a, b in zip(xa, xb)
                     if not (math.isnan(a) or math.isnan(b))]
            if len(pairs) < 10:
                print(f"  {m:<20} {'(fewer than 10 valid pairs — skip)':>33}")
                continue
            a_vals, b_vals = zip(*pairs)
            diffs = [a - b for a, b in zip(a_vals, b_vals)]
            if all(d == 0.0 for d in diffs):
                print(f"  {m:<20} {_mean(list(a_vals)):>10.3f} {_mean(list(b_vals)):>10.3f}"
                      f" {'identical':>10}")
                continue
            _, p = _wilcoxon(list(a_vals), list(b_vals), alternative="two-sided")
            sig = ("***" if p < 0.001 else
                   ("**"  if p < 0.01  else
                    ("*"   if p < 0.05  else "ns")))
            print(f"  {m:<20} {_mean(list(a_vals)):>10.3f} {_mean(list(b_vals)):>10.3f}"
                  f" {p:>10.4f}  {sig}")
        except Exception as exc:
            print(f"  {m:<20} error: {exc}")


def _write_csv(rows: List[dict], path: str):
    import csv
    if not rows:
        return
    # preserve insertion order so CSV columns are deterministic
    seen: dict = {}
    for r in rows:
        for k in r:
            seen[k] = None
    keys = list(seen)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-question results written to {path}")


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _print_summary(rows: List[dict]):
    if not rows:
        print("No results to summarise.")
        return
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
    ap = argparse.ArgumentParser(
        description="Evaluate the EWU RAG pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--make-template", metavar="PATH",
                    help="Write a starter gold JSONL template and exit.")
    ap.add_argument("--gold", metavar="PATH",
                    help="Gold JSONL file (from batch_eval.py) to evaluate against.")
    ap.add_argument("--out", default="results.csv",
                    help="Output CSV path (default: results.csv).")
    ap.add_argument("--no-ragas", action="store_true",
                    help="Skip RAGAS LLM-judged metrics.")
    ap.add_argument("--ablation", action="store_true",
                    help="Run all 4 ablation modes (requires --gold).")
    ap.add_argument("--ablation-prefix", default="ablation",
                    help="Filename prefix for per-mode CSVs (default: ablation).")
    ap.add_argument("--compare", nargs=2, metavar=("CSV_A", "CSV_B"),
                    help="Run Wilcoxon signed-rank test between two result CSVs.")
    ap.add_argument("--label-a", default="A", help="Label for first CSV in --compare.")
    ap.add_argument("--label-b", default="B", help="Label for second CSV in --compare.")
    ap.add_argument("--compare-metrics", nargs="+", metavar="METRIC",
                    help="Subset of metrics to test in --compare (default: all numeric).")
    args = ap.parse_args()

    if args.make_template:
        make_template(args.make_template)
    elif args.ablation:
        if not args.gold:
            ap.error("--ablation requires --gold PATH")
        run_ablation_study(args.gold, out_prefix=args.ablation_prefix)
    elif args.compare:
        wilcoxon_compare(
            args.compare[0], args.compare[1],
            label_a=args.label_a, label_b=args.label_b,
            metrics=args.compare_metrics,
        )
    elif args.gold:
        evaluate_system(args.gold, args.out, with_ragas=not args.no_ragas)
    else:
        ap.error("pass one of: --make-template PATH | --gold PATH | --ablation --gold PATH | --compare A.csv B.csv")
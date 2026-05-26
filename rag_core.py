import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    EMBED_MODEL, RERANKER_MODEL, LLM_MODEL,
    EMBED_DEVICE, RERANK_DEVICE,
    K_DENSE, K_SPARSE, RERANK_CANDIDATE, RERANK_TOP_K, RRF_K, CATEGORY_BOOST,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    VECTORSTORE_DIR, MAX_HISTORY_TURNS,
    GROQ_API_KEY,
)
from query_processing import expand_query

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Vector store + BM25
# ──────────────────────────────────────────────────────────────────
def build_vectorstore(docs: List[Document]) -> Tuple[FAISS, BM25Okapi, List[Document]]:
    log.info("Building embeddings with %s on %s…", EMBED_MODEL, EMBED_DEVICE)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs_path = Path(VECTORSTORE_DIR)
    if vs_path.exists():
        log.info("Loading cached FAISS index from %s", vs_path)
        vectorstore = FAISS.load_local(
            str(vs_path), embeddings, allow_dangerous_deserialization=True
        )
        all_docs = list(vectorstore.docstore._dict.values())
        log.info("Recovered %d docs from cached index docstore", len(all_docs))
    else:
        log.info("Building FAISS index from scratch (%d chunks)…", len(docs))
        vectorstore = FAISS.from_documents(docs, embeddings)
        vs_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vs_path))
        all_docs = docs
        log.info("FAISS index saved to %s", vs_path)

    from query_processing import _TOKEN_RE
    texts = [d.page_content for d in all_docs]
    bm25 = BM25Okapi([_TOKEN_RE.findall(t.lower()) for t in texts])
    return vectorstore, bm25, all_docs


# ──────────────────────────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        log.info("Loading reranker: %s on %s", model_name, RERANK_DEVICE)
        self.model = CrossEncoder(
            model_name,
            device=RERANK_DEVICE,
            max_length=1024,
            automodel_args={"torch_dtype": torch.bfloat16},
        )

    def rerank(self, query: str, docs: List[Document], top_k: int = RERANK_TOP_K):
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ──────────────────────────────────────────────────────────────────
# LLM (Groq)
# ──────────────────────────────────────────────────────────────────
def load_llm(model_id: str = LLM_MODEL):
    log.info("Initialising Groq client with model: %s", model_id)
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    client = Groq(api_key=GROQ_API_KEY)
    return client, model_id


# ──────────────────────────────────────────────────────────────────
# Fusion
# ──────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int = RRF_K):
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = f"{doc.page_content}||{doc.metadata.get('source', '')}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc
    return scores, doc_map

def debug_query(question: str, vectorstore, bm25, all_docs, reranker):
    """Call this directly to see exactly what the pipeline retrieves."""
    expanded = expand_query(question)
    print(f"\n{'='*60}")
    print(f"Original query: {question}")
    print(f"Expanded variants: {expanded}")

    # dense results
    print(f"\n--- Dense retrieval (top 3 per variant) ---")
    for q in expanded:
        results = vectorstore.similarity_search(q, k=3)
        print(f"\n  variant: '{q}'")
        for i, d in enumerate(results):
            print(f"  [{i+1}] source={d.metadata.get('source','')} | {d.page_content[:120]!r}")

    # sparse results
    print(f"\n--- BM25 sparse retrieval (top 3) ---")
    scores = bm25.get_scores(question.lower().split())
    top_idx = np.argsort(scores)[::-1][:3]
    for i in top_idx:
        print(f"  score={scores[i]:.3f} | {all_docs[i].page_content[:120]!r}")

    # after reranking
    print(f"\n--- After rerank (final context) ---")
    dense_lists = [vectorstore.similarity_search(q, k=K_DENSE) for q in expanded]
    sparse_scores = bm25.get_scores(question.lower().split())
    sparse_top = np.argsort(sparse_scores)[::-1][:K_SPARSE]
    sparse_lists = [[all_docs[i] for i in sparse_top]]
    fused_scores, doc_map = _reciprocal_rank_fusion(dense_lists + sparse_lists, RRF_K)
    ordered = sorted(doc_map, key=lambda k: fused_scores[k], reverse=True)
    candidates = [doc_map[k] for k in ordered[:RERANK_CANDIDATE]]
    reranked = reranker.rerank(question, candidates, top_k=RERANK_TOP_K)
    for i, (doc, score) in enumerate(reranked):
        print(f"  [{i+1}] rerank_score={score:.3f} source={doc.metadata.get('source','')} | {doc.page_content[:150]!r}")
    print('='*60)
# ──────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a helpful, knowledgeable assistant for East West University (EWU), "
    "Bangladesh. Answer using ONLY the provided context; never invent information.\n"
    "Match the language of the question:\n"
    "  - English question -> answer in English\n"
    "  - Bangla question -> answer in Bangla\n"
    "  - Banglish (Bangla written in English letters) -> answer in formal Bangla\n"
    "If the answer is not in the context, say exactly: "
    "\"I don't have enough information to answer that.\" "
    "(or in Bangla: \"আমার কাছে এই তথ্যটি নেই।\"). "
    "Do not list source URLs yourself; they are appended automatically."
)


# ──────────────────────────────────────────────────────────────────
# Chain
# ──────────────────────────────────────────────────────────────────
def build_rag_chain(llm, vectorstore, bm25, all_docs, reranker,
                    ablation_mode: str = "full"):
    """
    ablation_mode:
      "full"         — complete pipeline (default)
      "no_expansion" — raw query only, no expand_query
      "no_rerank"    — skip cross-encoder reranking, use RRF top-k directly
      "dense_only"   — dense retrieval only, no BM25 sparse lists
    """
    client, model_id = llm
    from query_processing import _TOKEN_RE

    _BOOSTS: List[Tuple[str, str, int]] = [
        ("faculty", "faculty", 3), ("teacher", "faculty", 3),
        ("fee", "tuition fees", 3), ("tuition", "tuition fees", 3),
        ("scholarship", "tuition fees", 2), ("waiver", "tuition fees", 2),
        ("deadline", "admission deadlines", 3), ("deadline", "admission process", 3),
        ("admission", "admission deadlines", 2), ("admission", "admission process", 2),
        ("vorti", "admission process", 2), ("vorti", "admission requirements", 2),
        ("requirement", "admission requirements", 3), ("eligib", "admission requirements", 3),
        ("cgpa", "admission requirements", 2),
        ("grade", "grading", 2), ("cgpa", "grading", 2), ("result", "grading", 2),
    ]

    def retrieve(question: str):
        expanded = [question] if ablation_mode == "no_expansion" else expand_query(question)
        q_lower = question.lower()

        dense_lists = [vectorstore.similarity_search(q, k=K_DENSE) for q in expanded]

        if ablation_mode == "dense_only":
            sparse_lists = []
        else:
            sparse_lists = []
            for q in expanded:
                scores = bm25.get_scores(_TOKEN_RE.findall(q.lower()))
                top_idx = np.argsort(scores)[::-1][:K_SPARSE]
                sparse_lists.append([all_docs[i] for i in top_idx])

        fused_scores, doc_map = _reciprocal_rank_fusion(dense_lists + sparse_lists, RRF_K)

        for key, doc in doc_map.items():
            cat = doc.metadata.get("category", "").lower()
            bonus = sum(pts for kw, cat_m, pts in _BOOSTS if kw in q_lower and cat_m in cat)
            if bonus:
                fused_scores[key] += bonus * CATEGORY_BOOST

        ordered = sorted(doc_map, key=lambda k: fused_scores[k], reverse=True)

        if ablation_mode == "no_rerank":
            context_docs = [doc_map[k] for k in ordered[:RERANK_TOP_K]]
        else:
            candidates = [doc_map[k] for k in ordered[:RERANK_CANDIDATE]]
            reranked = reranker.rerank(question, candidates, top_k=RERANK_TOP_K)
            context_docs = [doc for doc, _ in reranked]

        context_text = "\n\n".join(d.page_content for d in context_docs)
        sources = list(dict.fromkeys(
            d.metadata.get("source", "") for d in context_docs
            if d.metadata.get("source", "")
        ))
        return context_text, sources, [d.page_content for d in context_docs]

    def _call_groq(history: list, context: str, question: str) -> str:
        user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        turns = history[-MAX_HISTORY_TURNS:]

        messages = [{"role": "system", "content": _SYSTEM}]
        for u, a in turns:
            messages.append({"role": "user",      "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_content})

        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return resp.choices[0].message.content.strip()

    def chain(question: str, history=None) -> dict:
        history = history or []
        context_text, sources, contexts = retrieve(question)
        answer = _call_groq(history, context_text, question)
        answer = answer or "I don't have enough information to answer that."
        return {"answer": answer, "sources": sources, "contexts": contexts}

    return chain
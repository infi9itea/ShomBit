import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import transformers
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    EMBED_MODEL, RERANKER_MODEL, LLM_MODEL,
    K_DENSE, K_SPARSE, RERANK_CANDIDATE, RERANK_TOP_K, RRF_K, CATEGORY_BOOST,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY, DO_SAMPLE,
    VECTORSTORE_DIR, MAX_HISTORY_TURNS,
)
from query_processing import expand_query

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Vector store + BM25
# ──────────────────────────────────────────────────────────────────
def build_vectorstore(docs: List[Document]) -> Tuple[FAISS, BM25Okapi, List[Document]]:
    log.info("Building embeddings with %s…", EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs_path = Path(VECTORSTORE_DIR)
    if vs_path.exists():
        log.info("Loading cached FAISS index from %s", vs_path)
        vectorstore = FAISS.load_local(
            str(vs_path), embeddings, allow_dangerous_deserialization=True
        )
        # CRITICAL: use the docs actually inside the index so BM25 / sparse /
        # RRF all reference the identical document set.
        all_docs = list(vectorstore.docstore._dict.values())
        log.info("Recovered %d docs from cached index docstore", len(all_docs))
    else:
        log.info("Building FAISS index from scratch (%d chunks)…", len(docs))
        vectorstore = FAISS.from_documents(docs, embeddings)
        vs_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vs_path))
        all_docs = docs
        log.info("FAISS index saved to %s", vs_path)

    texts = [d.page_content for d in all_docs]
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    return vectorstore, bm25, all_docs


# ──────────────────────────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────────────────────────
class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        log.info("Loading reranker: %s", model_name)
        self.model = CrossEncoder(model_name, device="cuda:0")

    def rerank(self, query: str, docs: List[Document], top_k: int = RERANK_TOP_K):
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ──────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────
def load_llm(hf_token: str, model_id: str = LLM_MODEL) -> HuggingFacePipeline:
    log.info("Loading LLM: %s", model_id)
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,   # output is only the completion -> no marker parsing
    )
    if DO_SAMPLE:                 # sampling params are only valid when sampling
        gen_kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)

    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
    return HuggingFacePipeline(pipeline=pipe)


# ──────────────────────────────────────────────────────────────────
# Fusion + prompt
# ──────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int = RRF_K):
    """Return (scores_by_key, doc_by_key)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc
    return scores, doc_map


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


def _build_prompt(tokenizer, history: List[Tuple[str, str]], context: str, question: str) -> str:
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    turns = history[-MAX_HISTORY_TURNS:]
    messages = [{"role": "system", "content": _SYSTEM}]
    for u, a in turns:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some Mistral templates reject a system role: fold system into 1st user msg.
        conv, first = [], True
        for u, a in turns:
            conv.append({"role": "user", "content": (_SYSTEM + "\n\n" + u) if first else u})
            conv.append({"role": "assistant", "content": a})
            first = False
        conv.append({"role": "user", "content": (_SYSTEM + "\n\n" + user_content) if first else user_content})
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


# ──────────────────────────────────────────────────────────────────
# Chain
# ──────────────────────────────────────────────────────────────────
def build_rag_chain(llm, vectorstore, bm25, all_docs, reranker):
    """Returns chain(question, history=[]) -> {answer, sources, contexts}."""
    tokenizer = llm.pipeline.tokenizer

    _BOOSTS: List[Tuple[str, str, int]] = [
        ("faculty", "faculty", 3), ("teacher", "faculty", 3),
        ("fee", "tuition fees", 3), ("tuition", "tuition fees", 3),
        ("scholarship", "tuition fees", 2),
        ("deadline", "admission deadlines", 3), ("admission", "admission deadlines", 2),
        ("grade", "grading", 2), ("cgpa", "grading", 2),
    ]

    def retrieve(question: str):
        expanded = expand_query(question)
        q_lower = question.lower()

        dense_lists = [vectorstore.similarity_search(q, k=K_DENSE) for q in expanded]

        sparse_lists = []
        for q in expanded:
            scores = bm25.get_scores(q.lower().split())
            top_idx = np.argsort(scores)[::-1][:K_SPARSE]
            sparse_lists.append([all_docs[i] for i in top_idx])

        fused_scores, doc_map = _reciprocal_rank_fusion(dense_lists + sparse_lists, RRF_K)

        # additive category bonus (does not override semantic ranking)
        for key, doc in doc_map.items():
            cat = doc.metadata.get("category", "").lower()
            bonus = sum(pts for kw, cat_m, pts in _BOOSTS if kw in q_lower and cat_m in cat)
            if bonus:
                fused_scores[key] += bonus * CATEGORY_BOOST

        ordered = sorted(doc_map, key=lambda k: fused_scores[k], reverse=True)
        candidates = [doc_map[k] for k in ordered[:RERANK_CANDIDATE]]

        reranked = reranker.rerank(question, candidates, top_k=RERANK_TOP_K)
        context_docs = [doc for doc, _ in reranked]
        context_text = "\n\n".join(d.page_content for d in context_docs)

        # ordered, de-duplicated source list (preserves rank for retrieval metrics)
        sources = list(dict.fromkeys(
            d.metadata.get("source", "") for d in context_docs
            if d.metadata.get("source", "").startswith("http")
        ))
        return context_text, sources, [d.page_content for d in context_docs]

    def chain(question: str, history=None) -> dict:
        history = history or []
        context_text, sources, contexts = retrieve(question)
        prompt = _build_prompt(tokenizer, history, context_text, question)
        raw = llm.invoke(prompt)
        answer = raw.strip() or "I don't have enough information to answer that."
        return {"answer": answer, "sources": sources, "contexts": contexts}

    return chain
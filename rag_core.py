import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import transformers
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    EMBED_MODEL, RERANKER_MODEL, LLM_MODEL,
    K_DENSE, K_SPARSE, RERANK_CANDIDATE, RERANK_TOP_K,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    VECTORSTORE_DIR, MAX_HISTORY_TURNS,
)
from query_processing import expand_query

log = logging.getLogger(__name__)


def build_vectorstore(
    docs: List[Document],
) -> Tuple[FAISS, BM25Okapi, List[Document]]:
    log.info("Building embeddings with %s…", EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs_path = Path(VECTORSTORE_DIR)
    if vs_path.exists():
        log.info("Loading cached FAISS index from %s", vs_path)
        vectorstore = FAISS.load_local(str(vs_path), embeddings, allow_dangerous_deserialization=True)
    else:
        log.info("Building FAISS index from scratch…")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vs_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vs_path))
        log.info("FAISS index saved to %s", vs_path)

    texts = [d.page_content for d in docs]
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    return vectorstore, bm25, docs


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        log.info("Loading reranker: %s", model_name)
        self.model = CrossEncoder(model_name, device="cuda:0")

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = RERANK_TOP_K,
    ) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def load_llm(hf_token: str, model_id: str = LLM_MODEL) -> HuggingFacePipeline:
    log.info("Loading LLM: %s", model_id)
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipeline)


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Combine multiple ranked lists with RRF.
    Each doc's score = sum(1 / (k + rank)) across all lists it appears in.
    Returns a deduplicated list sorted by descending fused score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    merged = sorted(doc_map.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in merged]


_TEMPLATE = """\
<|system|>
You are a helpful and knowledgeable assistant for East West University (EWU), \
Bangladesh. Your sole job is to answer questions based on the provided context.

Rules:
- Use ONLY the provided context. Never invent information.
- Match the language of the question:
    • English question → English answer
    • Bangla question  → Bangla answer
    • Banglish question → formal Bangla answer
- If the answer is not in the context, say:
    "I don't have enough information to answer that."
    (or in Bangla: "আমার কাছে এই তথ্যটি নেই।")
- At the end of your answer, if sources are available, cite them concisely as:
    Sources: <url1>, <url2>
</s>

<|user|>
{history}

CONTEXT:
{context}

QUESTION:
{question}
</s>

<|assistant|>
"""

_PROMPT = PromptTemplate.from_template(_TEMPLATE)


def _format_history(history: List[Tuple[str, str]]) -> str:
    """Convert last N turns into a prompt-friendly string."""
    turns = history[-MAX_HISTORY_TURNS:]
    if not turns:
        return ""
    lines = ["CONVERSATION HISTORY:"]
    for user_msg, bot_msg in turns:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {bot_msg}")
    return "\n".join(lines) + "\n"


def build_rag_chain(
    llm: HuggingFacePipeline,
    vectorstore: FAISS,
    bm25: BM25Okapi,
    all_docs: List[Document],
    reranker: Reranker,
):
    """
    Returns a callable:
        chain(question, history=[]) → {"answer": str, "sources": list[str]}
    """

    _BOOSTS: List[Tuple[str, str, int]] = [
        ("faculty", "faculty", 3),
        ("teacher", "faculty", 3),
        ("fee", "tuition fees", 3),
        ("tuition", "tuition fees", 3),
        ("scholarship", "tuition fees", 2),
        ("deadline", "admission deadlines", 3),
        ("admission", "admission deadlines", 2),
        ("grade", "grading", 2),
        ("cgpa", "grading", 2),
    ]

    def retrieve(question: str) -> Tuple[str, List[str]]:
        expanded = expand_query(question)
        q_lower  = question.lower()

        dense_lists: List[List[Document]] = [
            vectorstore.similarity_search(q, k=K_DENSE) for q in expanded
        ]

        sparse_lists: List[List[Document]] = []
        for q in expanded:
            tokens = q.lower().split()
            scores = bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:K_SPARSE]
            sparse_lists.append([all_docs[i] for i in top_idx])

        # Merge with RRF
        fused = _reciprocal_rank_fusion(dense_lists + sparse_lists)

        # Category boost
        def _boost(doc: Document) -> int:
            cat = doc.metadata.get("category", "").lower()
            total = 0
            for kw, cat_match, pts in _BOOSTS:
                if kw in q_lower and cat_match in cat:
                    total += pts
            return total

        fused.sort(key=_boost, reverse=True)         

        reranked = reranker.rerank(question, fused[:RERANK_CANDIDATE], top_k=RERANK_TOP_K)

        context_docs  = [doc for doc, _ in reranked]
        context_text  = "\n\n".join(d.page_content for d in context_docs)
        sources       = list({
            d.metadata.get("source", "")
            for d in context_docs
            if d.metadata.get("source", "").startswith("http")
        })

        return context_text, sources

    def chain(question: str, history: List[Tuple[str, str]] | None = None) -> dict:
        history = history or []
        context_text, sources = retrieve(question)
        history_str = _format_history(history)

        prompt_value = _PROMPT.format(
            history=history_str,
            context=context_text,
            question=question,
        )

        raw = llm.invoke(prompt_value)

        marker = "<|assistant|>"
        idx = raw.rfind(marker)
        answer = raw[idx + len(marker):].strip() if idx != -1 else raw.strip()
        answer = answer or "I don't have enough information to answer that."

        return {"answer": answer, "sources": sources}

    return chain

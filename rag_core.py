import torch
import transformers
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from query_processing import expand_query
import numpy as np

def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    texts = [doc.page_content for doc in docs]
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return vectorstore, bm25, docs

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, device="cuda:0")

    def rerank(self, query, docs, top_k=3):
        if not docs: return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

def load_llm(hf_token, model_id="mistralai/Ministral-8B-Instruct-2410"):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
    pipeline = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=768, temperature=0.3, top_p=0.9, repetition_penalty=1.05   
    )
    return HuggingFacePipeline(pipeline=pipeline)

def build_rag_chain(llm, vectorstore, bm25, all_docs, reranker):

    def hybrid_retrieve(question, k_dense=15, k_sparse=10):

        expanded_queries = expand_query(question)

        dense_results = []

        for q in expanded_queries:
            dense_results.extend(
                vectorstore.similarity_search(q, k=k_dense)
            )

        bm25_scores = []

        for q in expanded_queries:
            tokenized_query = q.lower().split()
            scores = bm25.get_scores(tokenized_query)

            top_indices = np.argsort(scores)[::-1][:k_sparse]

            for idx in top_indices:
                bm25_scores.append(all_docs[idx])

        combined = dense_results + bm25_scores

        unique_docs = []
        seen = set()

        for doc in combined:
            content = doc.page_content

            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        boosted_docs = []

        for doc in unique_docs:

            category = doc.metadata.get("category", "").lower()

            boost = 0

            q_lower = question.lower()

            if "faculty" in q_lower and "faculty" in category:
                boost += 2

            if "fee" in q_lower and "tuition" in category:
                boost += 2

            boosted_docs.append((doc, boost))

        boosted_docs.sort(key=lambda x: x[1], reverse=True)

        docs_only = [d for d, _ in boosted_docs]

        reranked = reranker.rerank(question, docs_only[:15], top_k=8)

        return "\n\n".join([doc.page_content for doc in reranked])

    template = """
    <|system|>
    You are a helpful and knowledgeable assistant for East West University (EWU).

    Rules:
    - Use ONLY the provided context.
    - If the question is in English, answer in English.
    - If the question is in Bangla, answer in Bangla.
    - If the question is in Banglish, answer in formal Bangla script.
    - Do not hallucinate.
    - If unsure, say:
      "I don't have enough information to answer that."
      OR
      "আমার কাছে এই তথ্যটি নেই।"

    </s>

    <|user|>

    CONTEXT:
    {context}

    QUESTION:
    {question}

    </s>

    <|assistant|>
    """

    prompt = PromptTemplate.from_template(template)

    return (
        {
            "context": hybrid_retrieve,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
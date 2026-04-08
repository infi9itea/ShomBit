import torch
import transformers
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(docs, embeddings)

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

def build_rag_chain(llm, vectorstore, reranker):
    def retrieve_and_rerank(question):
        retrieved_docs = vectorstore.similarity_search(question, k=25)
        dynamic_priority = [d for d in retrieved_docs if "ewubd.edu" in d.metadata.get("source", "")]
        static_others = [d for d in retrieved_docs if "ewubd.edu" not in d.metadata.get("source", "")]
        docs_to_rerank = dynamic_priority if dynamic_priority else static_others
        reranked = reranker.rerank(question, docs_to_rerank[:8], top_k=8)
        return "\n\n".join([doc.page_content for doc in reranked])

    template = """
    <|system|>
    You are a helpful and knowledgeable assistant for East West University (EWU).
    Use only the provided context to answer questions. Answer in the same language the user asks in (English, standard Bangla, or Banglish).
    If unsure, say: "I don't have enough information to answer that." / "আমার কাছে এই তথ্যটি নেই।"
    </s>
    <|user|>
    CONTEXT: {context}
    QUESTION: {question}
    </s>
    <|assistant|>
    """
    prompt = PromptTemplate.from_template(template)
    return ({"context": retrieve_and_rerank, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
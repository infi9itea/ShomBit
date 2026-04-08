import os
import gradio as gr
from pyngrok import ngrok
from huggingface_hub import login
from data_pipeline import load_json_docs, chunk_documents, scrape_dynamic_docs
from rag_core import build_vectorstore, Reranker, load_llm, build_rag_chain

HF_TOKEN = os.environ.get("HF_TOKEN")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

login(HF_TOKEN)

print("Starting Data Pipeline...")
all_docs = chunk_documents(load_json_docs(DATA_DIR)) + chunk_documents(scrape_dynamic_docs())

print("Loading Models...")
vectorstore = build_vectorstore(all_docs)
reranker = Reranker()
llm = load_llm(HF_TOKEN)
rag_chain = build_rag_chain(llm, vectorstore, reranker)

def answer_fn(user_question):
    try:
        response = rag_chain.invoke(user_question)
        assistant_start = response.find("<|assistant|>")
        if assistant_start != -1:
            cleaned = response[assistant_start + len("<|assistant|>"):].strip()
            return cleaned if cleaned else "I don't have enough information to answer that question."
        return response.strip() or "I don't have enough information to answer that question."
    except Exception as e:
        return f" Error: {e}"

if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(7860).public_url
    print(f"\n YOUR WEBSITE API ENDPOINT: {public_url}\n")

# 5. Launch API Server
iface = gr.Interface(
    fn=answer_fn,
    inputs=gr.Textbox(lines=2),
    outputs=gr.Textbox(lines=15),
    title="EWU RAG API"
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
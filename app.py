import logging
import sys
from config import HF_TOKEN, NGROK_TOKEN, DATA_DIR   # GROQ_API_KEY not needed here, rag_core reads it directly
from rag_core import build_vectorstore, Reranker, load_llm, build_rag_chain
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from huggingface_hub import login
from config import HF_TOKEN, NGROK_TOKEN, DATA_DIR

if not HF_TOKEN:
    log.warning("HF_TOKEN is not set — embedding and reranker models may fail.")
    sys.exit(1)
login(HF_TOKEN)

from data_pipeline import load_json_docs, chunk_documents, scrape_dynamic_docs
from rag_core import build_vectorstore, Reranker, load_llm, build_rag_chain

log.info("=== Starting EWU RAG Pipeline ===")

log.info("Step 1/4  Loading and chunking documents…")
json_docs = chunk_documents(load_json_docs(DATA_DIR))
scraped_docs = chunk_documents(scrape_dynamic_docs())
all_docs = json_docs + scraped_docs
log.info("Total chunks: %d", len(all_docs))

log.info("Step 2/4  Building vectorstore…")
vectorstore, bm25, all_docs = build_vectorstore(all_docs)

log.info("Step 3/4  Loading reranker and LLM…")
reranker = Reranker()
llm = load_llm()

log.info("Step 4/4  Assembling RAG chain…")
rag_chain = build_rag_chain(llm, vectorstore, bm25, all_docs, reranker)

# ── TEMPORARY DEBUG — remove after diagnosis ──────────────────────
from rag_core import debug_query
TEST_QUERIES = [
    "CSE tuition fee koto?",        # put your failing queries here
    "What is the grading system?",
    "vorti deadline kobe?",
]
for q in TEST_QUERIES:
    debug_query(q, vectorstore, bm25, all_docs, reranker)
# ─────────────────────────────────────────────────────────────────

def chat_fn(user_message: str, history: list[dict]):
    """history is a list of {"role", "content"} dicts (Gradio type='messages')."""
    if not user_message.strip():
        return "", history

    # reconstruct (user, assistant) pairs for the LLM's short-term memory
    internal, pending = [], None
    for m in history:
        if m["role"] == "user":
            pending = m["content"]
        elif m["role"] == "assistant" and pending is not None:
            internal.append((pending, m["content"]))
            pending = None

    try:
        result = rag_chain(user_message, history=internal)
        answer = result["answer"]
        sources = result["sources"]
        if sources:
            answer += "\n\n📚 **Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    except Exception as exc:
        log.exception("Error during inference")
        answer = f"⚠️ An error occurred: {exc}"

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return "", history


import gradio as gr

with gr.Blocks(title="EWU Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎓 East West University Assistant
        Ask anything about EWU — admissions, fees, faculty, grading, events, and more.
        Questions in **English**, **Bangla**, or **Banglish** are all supported.
        """
    )

    chatbot = gr.Chatbot(
        label="EWU Assistant",
        type="messages",
        height=520,
        show_copy_button=True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your question here… (e.g. 'CSE vorti fee koto?')",
            lines=2, scale=9, show_label=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear conversation")

    gr.Examples(
        examples=[
            "What is the tuition fee for CSE?",
            "CSE vorti deadline kobe?",
            "What are the grading rules at EWU?",
            "Who are the faculty members of the CSE department?",
            "কম্পিউটার বিজ্ঞান বিভাগে ভর্তির যোগ্যতা কি?",
        ],
        inputs=msg_box,
    )

    send_btn.click(chat_fn, [msg_box, chatbot], [msg_box, chatbot])
    msg_box.submit(chat_fn, [msg_box, chatbot], [msg_box, chatbot])
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg_box])


if NGROK_TOKEN:
    from pyngrok import ngrok as _ngrok
    _ngrok.set_auth_token(NGROK_TOKEN)
    public_url = _ngrok.connect(7860).public_url
    log.info("Public URL: %s", public_url)

log.info("Launching Gradio server on 0.0.0.0:7860")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
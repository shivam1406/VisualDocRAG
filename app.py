
import streamlit as st
from src.pipeline import VisualDocRAG
from src.config import SETTINGS
import os

st.set_page_config(page_title="Visual Document Analysis RAG", layout="wide")
st.title("📄🔎 Visual Document Analysis RAG")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=SETTINGS.top_k)
    st.markdown("---")
    st.write("Embedding model:", SETTINGS.openrouter_model)
    st.write("Chroma dir:", SETTINGS.persist_dir)
    st.write("Collection:", SETTINGS.collection_name)
    st.write("OpenRouter Enabled:", "Yes" if SETTINGS.openrouter_api_key else "No")

if "rag" not in st.session_state:
    st.session_state["rag"] = VisualDocRAG()
rag = st.session_state["rag"]

st.subheader("1) Ingest Documents")
files = st.file_uploader("Upload PDFs or Images", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)
if files and st.button("Ingest"):
    for f in files:
        tmp = os.path.join("data","samples", f.name)
        os.makedirs("data/samples", exist_ok=True)
        with open(tmp, "wb") as out:
            out.write(f.read())
        with st.spinner(f"Ingesting {f.name}..."):
            res = rag.ingest_file(tmp)
        if res.get("ok"):
            st.success(f"{f.name}: {res['message']} (latency {res['latency_ms']} ms)")
        else:
            st.error(f"{f.name}: {res['message']} (latency {res['latency_ms']} ms)")

st.subheader("2) Ask a Question")
q = st.text_input("Your question")
if st.button("Search and Answer") and q.strip():
    with st.spinner("Retrieving..."):
        res = rag.query(q, top_k=top_k)
    st.markdown("### Answer")
    st.write(res["answer"])
    st.markdown("### Retrieved Contexts")
    for i, ctx in enumerate(res["contexts"]):
        meta = ctx["metadata"]
        with st.expander(f"Context {i+1} — p.{meta.get('page','?')} / {meta.get('modality','?')} (score≈{ctx.get('score',0):.3f})"):
            st.code(ctx["text"])
    st.caption(f"Latency: {res['latency_ms']} ms")

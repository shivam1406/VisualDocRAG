# 📄 VisualDocRAG — Retrieval-Augmented Generation for Visual Documents - Check at(https://visualdocrag.streamlit.app/)

VisualDocRAG is a Retrieval-Augmented Generation (RAG) pipeline that processes **PDFs, scanned images, and charts**, extracts text via OCR or native text extraction, and enables question answering using a vector database.

This project was built to demonstrate **document ingestion, semantic search, and LLM-based answering**.

---

## ✨ Features
- 📂 **Multi-format ingestion** — PDF (native & scanned), JPG, PNG
- 🔍 **Text + OCR extraction** (PyMuPDF & Tesseract)
- ✂ **Chunking & metadata tagging**
- 📦 **Vector store with ChromaDB** (persistent storage)
- 🤖 **LLM-based answer generation** (OpenRouter / OpenAI API)
- 🖥 **Streamlit-based UI**
- 📊 **Evaluation script** with latency metrics and RAGAS-ready JSON

---

## 📦 Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed & added to PATH (required for scanned docs)
- An **OpenRouter** or **OpenAI API key** for abstractive answers

---

## ⚙ Installation

```bash
# 1. Clone the repository
git clone https://github.com/ss.shivam1406/visual-doc-rag.git
cd visual-doc-rag

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📄 Environment Variables
### Create a ```.env``` file in the root:
```bash
# API keys (use one of these)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openrouter/your-model
```

---

## 🚀 Running the App
```bash
streamlit run app.py
```
Then:
- Upload PDFs/images via the Ingest File section.
- Run queries in the Ask a Question section.

---

## 📊 Running Evaluation
We include ```evaluate.py``` to run automated queries and export RAGAS-ready results:
```bash
python evaluate.py
```

- If ```.chroma/``` already exists, it will use the same ingested data from the frontend.
- If not, you can modify ```evaluate.py``` to auto-ingest test data before evaluation.

Output:
- ```evaluation_results.json``` → contains query, predicted answer, gold answer, contexts, and latency.

---

## 🧱 Project Structure
```
visual-doc-rag/
│
├── app.py                 # Streamlit UI
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
├── README.md
│
└── src/
    ├── config.py          # Settings loader
    ├── pipeline.py        # Main VisualDocRAG orchestrator
    ├── loaders.py         # PDF/image loaders (OCR & text)
    ├── chunking.py        # Chunking logic
    ├── vectorstore.py     # ChromaDB vector store
    ├── retriever.py       # Top-K semantic retriever
    ├── generator.py       # Answer generation (OpenRouter OpenAI/local)
    └── utils.py           # Helper functions
```

---

## 🛠 Troubleshooting

-**Missing imports in VS Code**: Make sure you select the ```.venv``` interpreter and install all dependencies.

-**TesseractNotFoundError**: Install Tesseract OCR and add it to PATH.

-**Quota errors**: Switch from OpenAI API to OpenRouter or update your billing plan.

-**Evaluation returns no answers**: Ensure ```.chroma/``` exists (ingest docs first).

---

## 🤝 Contribute
Pull requests welcome! For major changes, open an issue first to discuss what you’d like to change.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Shivam**  
📧 [ss.shivam1406@gmail.com](mailto:ss.shivam1406@gmail.com)  
🔗 [LinkedIn – ssshivam1406](https://www.linkedin.com/in/ssshivam1406/)

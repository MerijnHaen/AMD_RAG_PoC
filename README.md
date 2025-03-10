# AMD RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) pipeline that combines PubMed-based semantic search and GPT-powered synthesis for executive-level summaries on age-related macular degeneration (AMD).

## 🧠 Features
- Semantic search with FAISS + MiniLM
- GPT-3.5-turbo synthesis layer
- Streamlit UI for interactive querying
- Easily deployable on Streamlit Cloud

## 📦 Setup

```bash
pip install -r requirements.txt
streamlit run amd_rag_assistant.py
```

## 🔐 Environment Variables
Add your OpenAI key to `.streamlit/secrets.toml` or use Streamlit Cloud Secrets Manager:

```toml
OPENAI_API_KEY = "your-key-here"
```

## 📂 Required Files
Ensure the following files are present in the project root:
- `amd_vector_index.faiss`
- `amd_paper_metadata.json`

## 📤 Deployment
Deploy on Streamlit Cloud: [https://streamlit.io/cloud](https://streamlit.io/cloud)
# AMD_RAG_PoC

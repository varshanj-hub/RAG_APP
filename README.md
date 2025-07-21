# 📄 RAG_APP – Retrieval-Augmented Generation Q&A Assistant

RAG_APP is a user-friendly and visually clean Streamlit app for document-based question answering using Retrieval-Augmented Generation (RAG). It allows users to upload documents, chunk them intelligently, embed the content, and ask questions using advanced LLMs like Gemini or Groq.


---

## 🚀 1. Features

- 📥 Upload individual documents or ZIP archives
- 📄 Supports PDF, Word, PPTX, and plain text
- ✂️ Semantic chunking with overlap and sentence control
- 🔍 Embedding using HuggingFace + ChromaDB
- 🤖 Ask document-related questions via Gemini or Groq LLMs
- 🎛️ Sidebar for chunking & model config
- 🎨 Professional UI using Streamlit with clear styling

---

## 📦 2. Installation

1. **Clone the repository**

```bash
git clone https://github.com/varshanj-hub/RAG_APP.git
cd RAG_APP
```


2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```


3. **Install all dependencies**

```
pip install -r req.txt
```


4. **Download required NLTK tokenizer**

```bash
# Run this once in Python shell or script
import nltk
nltk.download('punkt')
```


5. **Run the app**

```bash
streamlit run rag_app.py
```








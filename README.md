
# 💬 PDFChat AI

An AI-powered application that lets you chat with any PDF document using RAG (Retrieval Augmented Generation).

## 🎯 What it does
Upload any PDF and ask questions in natural language. The AI answers strictly from your document with source page references.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** LLaMA 3 via Groq API (free)
- **Vector DB:** FAISS
- **Embeddings:** TF-IDF + Scikit-learn
- **PDF Reading:** PyPDF2
- **Language:** Python 3.11


## 🚀 How to Run

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pdfchat-ai
cd pdfchat-ai
```

2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your Groq API key in `rag_pipeline.py`

5. Run the app
```bash
streamlit run app.py
```

## 📸 Features
- Upload any PDF document
- Real-time progress bar during processing
- Chat interface with history
- Source page citations
- Works with any domain (medical, legal, educational, etc.)

## 🔑 Get Free API Key
Get your free Groq API key at: https://console.groq.com
=======


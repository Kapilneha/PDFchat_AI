import streamlit as st
import os
import tempfile
from rag_pipeline import (
    load_pdf,
    chunk_text,
    store_in_faiss,
    load_faiss_index,
    ask_question
)

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="PDFChat AI",
    page_icon="💬",
    layout="centered"
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stats-box {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        text-align: center;
    }
    .source-badge {
        background-color: #2d2d3f;
        border-radius: 5px;
        padding: 3px 8px;
        margin: 2px;
        display: inline-block;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================
st.markdown("""
    <div class="main-header">
        <h1>💬 PDFChat AI</h1>
        <p style="color: gray;">Upload any PDF and chat with it instantly using AI</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF — research papers, textbooks, manuals, reports!"
    )

    if uploaded_file is not None:
        # Show file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📎 **{uploaded_file.name}**\n\nSize: {file_size:.1f} MB")

        if st.button("🔄 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("⏳ Processing your PDF..."):
                try:
                    progress = st.progress(0)
                    status = st.empty()

                    # Save temp file
                    status.text("📥 Loading PDF...")
                    progress.progress(20)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Load PDF
                    pages = load_pdf(tmp_path)
                    progress.progress(40)

                    # Chunk
                    status.text("✂️ Chunking document...")
                    chunks = chunk_text(pages)
                    progress.progress(60)

                    # Embed + Store
                    status.text("🔢 Creating embeddings...")
                    store_in_faiss(chunks)
                    progress.progress(80)

                    # Load index
                    status.text("💾 Loading index...")
                    index, chunks_data, vectorizer = load_faiss_index()
                    progress.progress(100)

                    # Clean up
                    os.unlink(tmp_path)
                    status.text("✅ Done!")

                    # Save to session
                    st.session_state.index = index
                    st.session_state.chunks = chunks_data
                    st.session_state.vectorizer = vectorizer
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.total_pages = len(pages)
                    st.session_state.total_chunks = len(chunks)
                    st.session_state.messages = []

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")

    # Show stats if PDF processed
    if "pdf_processed" in st.session_state and st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state.total_pages)
        with col2:
            st.metric("Chunks", st.session_state.total_chunks)

        if st.button("🗑️ Clear & Upload New", use_container_width=True):
            st.session_state.pdf_processed = False
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("1. 📄 Upload any PDF")
    st.markdown("2. 🔄 Click Process PDF")
    st.markdown("3. 💬 Ask any question!")
    st.markdown("4. 🤖 AI answers from your doc")
    st.markdown("---")
    st.markdown("Built with **RAG + LLaMA 3** 🤖")
    st.markdown("*Powered by Groq + FAISS*")

# =============================================
# MAIN CHAT AREA
# =============================================
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome screen
if not st.session_state.pdf_processed:
    st.markdown("### 👋 Welcome to PDFChat AI!")
    st.markdown("Upload any PDF from the sidebar and start chatting with it!")

    st.markdown("---")
    st.markdown("### 💡 What can you upload?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📚 **Study**")
        st.markdown("Textbooks")
        st.markdown("Research papers")
        st.markdown("Lecture notes")
    with col2:
        st.markdown("💼 **Work**")
        st.markdown("Company reports")
        st.markdown("Policy documents")
        st.markdown("Project briefs")
    with col3:
        st.markdown("⚕️ **Medical**")
        st.markdown("Health guides")
        st.markdown("Drug manuals")
        st.markdown("Clinical reports")

    st.markdown("---")
    st.markdown("### 💡 Example Questions You Can Ask:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Summarize this document")
        st.markdown("- What is the main topic?")
        st.markdown("- List the key points")
    with col2:
        st.markdown("- What does page 1 say?")
        st.markdown("- Explain [any term] from the doc")
        st.markdown("- What conclusions are made?")

else:
    # Active chat
    st.success(f"📄 Chatting with: **{st.session_state.pdf_name}**")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Show suggested questions if no chat yet
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try asking:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Summarize this document"):
                st.session_state.suggested = "Summarize this document"
                st.rerun()
            if st.button("🔑 What are the key points?"):
                st.session_state.suggested = "What are the key points?"
                st.rerun()
        with col2:
            if st.button("📖 What is the main topic?"):
                st.session_state.suggested = "What is the main topic?"
                st.rerun()
            if st.button("📝 What conclusions are made?"):
                st.session_state.suggested = "What conclusions are made?"
                st.rerun()

    # Handle suggested question clicks
    if "suggested" in st.session_state and st.session_state.suggested:
        question = st.session_state.suggested
        st.session_state.suggested = None
    else:
        question = st.chat_input("Ask anything about your document...")

    if question:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    answer, results = ask_question(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.vectorizer
                    )

                    st.markdown(answer)

                    # Show sources nicely
                    pages_found = list(set([r['page_number'] for r in results]))
                    st.markdown("---")
                    source_text = " ".join([f"📄 Page {p}" for p in pages_found])
                    st.markdown(f"**📚 Sources:** {source_text}")

                    full_response = f"{answer}\n\n**📚 Sources:** {source_text}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                except Exception as e:
                    st.error(f"❌ Something went wrong: {str(e)}")
                    st.info("💡 Try re-processing the PDF from the sidebar.")
import PyPDF2
import numpy as np
import faiss
import pickle
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


# =============================================
# STEP 1: LOAD PDF
# =============================================
def load_pdf(file_path):
    pages = []
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            if text and text.strip():
                pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })
    print(f"✅ Loaded {len(pages)} pages from PDF")
    return pages


# =============================================
# STEP 2: CHUNK TEXT
# =============================================
def chunk_text(pages, chunk_size=500, chunk_overlap=100):
    chunks = []
    chunk_id = 0
    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk
                })
                chunk_id += 1
            start += chunk_size - chunk_overlap
    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# =============================================
# STEP 3: CREATE EMBEDDINGS
# =============================================
def get_embeddings(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=384)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    return matrix, vectorizer


# =============================================
# STEP 4: STORE IN FAISS
# =============================================
def store_in_faiss(chunks):
    print("⏳ Creating embeddings and storing in FAISS...")
    texts = [c["text"] for c in chunks]
    embeddings, vectorizer = get_embeddings(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    os.makedirs("faiss_db", exist_ok=True)
    faiss.write_index(index, "faiss_db/index.faiss")
    with open("faiss_db/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("faiss_db/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Stored {len(chunks)} chunks in FAISS")
    return index, chunks, vectorizer


# =============================================
# STEP 5: LOAD FAISS FROM DISK
# =============================================
def load_faiss_index():
    print("📂 Loading FAISS index from disk...")
    faiss_index = faiss.read_index("faiss_db/index.faiss")
    with open("faiss_db/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("faiss_db/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print(f"✅ Loaded index with {faiss_index.ntotal} chunks")
    return faiss_index, chunks, vectorizer


# =============================================
# STEP 6: SEARCH FAISS
# =============================================
def search(query, index, chunks, vectorizer, top_k=4):
    query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    scores, indices = index.search(query_vec, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "text": chunks[idx]["text"],
                "page_number": chunks[idx]["page_number"],
                "score": scores[0][i]
            })
    return results


# =============================================
# STEP 7: ASK QUESTION (RAG + GROQ)
# =============================================
def ask_question(question, index, chunks, vectorizer):

    # Retrieve relevant chunks
    print(f"\n🔍 Searching for: '{question}'")
    results = search(question, index, chunks, vectorizer, top_k=4)
    print(f"📄 Found {len(results)} relevant chunks")

    # Build context
    context = ""
    for result in results:
        context += f"\n[Page {result['page_number']}]\n{result['text']}\n"

    # Build prompt
    prompt = f"""You are a helpful medical information assistant.
Answer the question below using ONLY the context provided.
If the answer is not in the context, say "I couldn't find that information in the document."

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

ANSWER:"""

    # Call Groq
    print("🤖 Asking Groq (LLaMA 3)...")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    return answer, results


# =============================================
# MAIN - TEST IT
# =============================================
if __name__ == "__main__":

    # Load saved FAISS index (already built on Day 3!)
    index, chunks, vectorizer = load_faiss_index()

    # Test questions
    questions = [
        "What are the protein requirements for adults?",
        "What foods are rich in calcium?",
        "How many calories should a person consume daily?"
    ]

    for question in questions:
        print("\n" + "="*60)
        answer, results = ask_question(
            question, index, chunks, vectorizer
        )
        print(f"\n❓ Question: {question}")
        print(f"\n💬 Answer: {answer}")
        print(f"\n📚 Sources: Pages {[r['page_number'] for r in results]}")
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

st.set_page_config(page_title="AI PDF Q&A", layout="centered")
st.title("📄 AI Document Q&A Chatbot")
st.caption("Hỏi xoáy đáp xoay")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128
    )

embed_model = load_embedding_model()
llm = load_llm()

def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text):
    sentences = text.split(".")
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def retrieve_context(question, chunks, embeddings, threshold=0.3):
    q_emb = embed_model.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score < threshold:
        return None, best_score

    return chunks[best_idx], best_score

def generate_answer(question, context):
    prompt = f"""
    Answer the question based on the context below.
    Do not copy the sentence exactly.
    Answer clearly and naturally.

    Context: {context}
    Question: {question}
    Answer:
    """
    result = llm(prompt)
    return result[0]["generated_text"]

def generate_quiz(context):
    prompt = f"""
    Create ONE multiple-choice question for students based on the context below.

    Rules:
    - Provide 1 question
    - Provide 4 options (A, B, C, D)
    - Clearly mark the correct answer

    Context: {context}

    Format:
    Question:
    A.
    B.
    C.
    D.
    Correct answer:
    """
    result = llm(prompt)
    return result[0]["generated_text"]

uploaded_file = st.file_uploader("📤 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Đang đọc PDF..."):
        text = load_pdf(uploaded_file)

    st.write("🔢 Text length:", len(text))

    if len(text.strip()) == 0:
        st.error("PDF này không có text (có thể là scan ảnh).")
        st.stop()

    chunks = chunk_text(text)
    st.write("Số lượng chunks:", len(chunks))

    with st.spinner("🧠 Xây dựng embeddings..."):
        embeddings = embed_model.encode(chunks)

    st.success("✅ AI đã sẵn sàng!")

    tab1, tab2 = st.tabs(["❓ Hỏi câu hỏi", "📝 Câu đố"])

    with tab1:
        question = st.text_input("❓ Hỏi câu hỏi đi")

        if question:
            context, score = retrieve_context(question, chunks, embeddings)

            st.write("📊 Similarity score:", round(float(score), 3))

            if context is None:
                st.warning("❌ Không tìm thấy thông tin trong tài liệu.")
            else:
                st.subheader("💡 Trả lời bằng AI")
                answer = generate_answer(question, context)
                st.write(answer)

                with st.expander("📄 Trích nguồn"):
                    st.write(context)

    with tab2:
        st.write("📝 Sinh ra một quiz")

        if st.button("🎯 Sinh Quiz"):
            context = np.random.choice(chunks)
            quiz = generate_quiz(context)

            st.subheader("Quiz Question")
            st.write(quiz)

            with st.expander("📄 Source sentence"):
                st.write(context)

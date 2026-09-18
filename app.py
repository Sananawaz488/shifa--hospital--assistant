import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
 
# ---------------------------
# Config
# ---------------------------
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4
 
st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="centered")
 
# ---------------------------
# API key from secrets (never shown in UI / not hardcoded)
# ---------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your Streamlit secrets (Settings → Secrets).")
    st.stop()
 
client = Groq(api_key=GROQ_API_KEY)
 
 
# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
 
 
vectorstore = load_vectorstore()
 
 
# ---------------------------
# Helpers
# ---------------------------
def retrieve_chunks(query, k=TOP_K):
    results = vectorstore.similarity_search(query, k=k)
    return results
 
 
def build_context(chunks):
    context_parts = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("path", doc.metadata.get("source", "unknown"))
        context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
    return "\n\n".join(context_parts)
 
 
def generate_answer(question, chunks):
    context = build_context(chunks)
 
    system_prompt = (
        "You are a hospital policy assistant. Answer the user's question using ONLY "
        "the provided document excerpts below. If the answer is not contained in the "
        "excerpts, say you don't have that information in the available policy documents. "
        "Be clear, concise, and accurate. Do not make up information."
    )
 
    user_prompt = f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"
 
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
 
 
# ---------------------------
# UI
# ---------------------------
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask a question about hospital policies. Answers are grounded in your uploaded documents.")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
 
# Chat input
question = st.chat_input("Ask about a hospital policy...")
 
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating answer..."):
            chunks = retrieve_chunks(question)
 
            if not chunks:
                answer = "I couldn't find any relevant information in the knowledge base."
                sources = []
            else:
                answer = generate_answer(question, chunks)
                sources = sorted(set(
                    doc.metadata.get("path", doc.metadata.get("source", "unknown"))
                    for doc in chunks
                ))
 
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
 

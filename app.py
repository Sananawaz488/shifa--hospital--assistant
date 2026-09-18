import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq


# =========================================================
# CONFIG
# =========================================================

INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4


st.set_page_config(
    page_title="Shifa Hospital Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}


/* =========================
   HEADER
   ========================= */

.hospital-header {
    background: linear-gradient(135deg, #ffffff, #eef7ff);
    padding: 28px 32px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    margin-bottom: 22px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}

.hospital-title {
    font-size: 36px;
    font-weight: 750;
    color: #172033;
    margin-bottom: 6px;
}

.hospital-subtitle {
    font-size: 16px;
    color: #667085;
}

.status-box {
    background: #ecfdf3;
    border: 1px solid #abefc6;
    color: #067647;
    padding: 11px 15px;
    border-radius: 12px;
    font-size: 14px;
    margin-top: 16px;
}


/* =========================
   SIDEBAR
   ========================= */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 700;
    color: #172033;
}

.sidebar-text {
    color: #667085;
    font-size: 14px;
    line-height: 1.6;
}


/* =========================
   QUICK QUESTIONS
   ========================= */

.quick-title {
    font-size: 18px;
    font-weight: 650;
    color: #344054;
    margin-top: 12px;
    margin-bottom: 12px;
}


/* =========================
   CHAT
   ========================= */

div[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 10px;
    margin-bottom: 10px;
}


/* =========================
   SOURCE CARDS
   ========================= */

.source-card {
    background: #f8fafc;
    border: 1px solid #e4e7ec;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 7px;
    color: #475467;
    font-size: 14px;
}


/* =========================
   FOOTER
   ========================= */

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 12px;
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #eaecf0;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# API KEY
# =========================================================

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. Add it to your Streamlit secrets."
    )
    st.stop()


client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

@st.cache_resource(show_spinner="Loading hospital knowledge base...")
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


vectorstore = load_vectorstore()


# =========================================================
# RETRIEVE DOCUMENT CHUNKS
# =========================================================

def retrieve_chunks(query, k=TOP_K):

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(chunks):

    context_parts = []

    for i, doc in enumerate(chunks, start=1):

        source = doc.metadata.get(
            "path",
            doc.metadata.get(
                "source",
                "Unknown document"
            )
        )

        context_parts.append(
            f"[Document {i} - {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, chunks):

    context = build_context(chunks)

    system_prompt = """
You are a professional hospital policy assistant.

Answer the user's question ONLY using the provided hospital
policy document excerpts.

Rules:

1. Do not invent information.
2. If the answer is not available in the provided documents,
   clearly say that the information is not available in the
   current hospital policy documents.
3. Keep answers clear and concise.
4. Use bullet points when helpful.
5. Do not provide medical diagnosis or treatment advice.
"""

    user_prompt = f"""
Hospital Policy Documents:

{context}

User Question:

{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🏥 Shifa Hospital</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
<div class="sidebar-text">

<b>Hospital Policy Assistant</b>

<br><br>

Ask questions about hospital policies,
procedures and guidelines.

<br><br>

Answers are generated from the
hospital's uploaded policy documents.

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📚 Knowledge Base")

    st.success("Knowledge base loaded")

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.caption(
        "🔒 API key is securely stored in Streamlit Secrets."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
<div class="hospital-header">

    <div class="hospital-title">
        🏥 Shifa Hospital Assistant
    </div>

    <div class="hospital-subtitle">
        Your AI assistant for hospital policies,
        procedures and guidelines.
    </div>

    <div class="status-box">
        🟢 Knowledge base connected •
        Answers grounded in uploaded hospital documents
    </div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# QUICK QUESTIONS
# =========================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="quick-title">💡 Try asking</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🕐 Visiting hours?",
            use_container_width=True
        ):

            st.session_state.quick_question = (
                "What are the visiting hours for patients?"
            )

            st.rerun()

    with col2:

        if st.button(
            "📋 Admission policy?",
            use_container_width=True
        ):

            st.session_state.quick_question = (
                "What is the hospital admission policy?"
            )

            st.rerun()

    with col3:

        if st.button(
            "👨‍👩‍👧 Visitor policy?",
            use_container_width=True
        ):

            st.session_state.quick_question = (
                "What are the rules for hospital visitors?"
            )

            st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and msg.get("sources")
        ):

            with st.expander("📚 View Sources"):

                for source in msg["sources"]:

                    st.markdown(
                        f"""
<div class="source-card">
📄 {source}
</div>
""",
                        unsafe_allow_html=True
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask about a hospital policy..."
)


# =========================================================
# QUICK QUESTION PROCESSING
# =========================================================

if "quick_question" in st.session_state:

    question = st.session_state.quick_question

    del st.session_state.quick_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -------------------------
    # USER MESSAGE
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------
    # ASSISTANT MESSAGE
    # -------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching hospital policies..."
        ):

            chunks = retrieve_chunks(question)

            if not chunks:

                answer = (
                    "I couldn't find relevant information "
                    "in the available hospital policy documents."
                )

                sources = []

            else:

                answer = generate_answer(
                    question,
                    chunks
                )

                sources = sorted(
                    set(
                        doc.metadata.get(
                            "path",
                            doc.metadata.get(
                                "source",
                                "Unknown document"
                            )
                        )
                        for doc in chunks
                    )
                )


        st.markdown(answer)


        # -------------------------
        # SOURCES
        # -------------------------

        if sources:

            with st.expander("📚 View Sources"):

                for source in sources:

                    st.markdown(
                        f"""
<div class="source-card">
📄 {source}
</div>
""",
                        unsafe_allow_html=True
                    )


    # -------------------------
    # SAVE ASSISTANT MESSAGE
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
    Shifa Hospital Policy Assistant •
    AI-powered document-based assistance
</div>
""",
    unsafe_allow_html=True
)

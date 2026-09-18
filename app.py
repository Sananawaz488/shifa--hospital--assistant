import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq


# =========================================================
# CONFIG
# =========================================================

INDEX_DIR = "faiss_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keep your current Groq model
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5


# =========================================================
# PAGE CONFIG
# =========================================================

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
    background: #f6f8fb;
}

/* Main container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ================= HEADER ================= */

.hospital-header {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

.hospital-title {
    font-size: 32px;
    font-weight: 750;
    color: #172033;
    margin: 0;
}

.hospital-subtitle {
    margin-top: 7px;
    font-size: 15px;
    color: #667085;
}

.status-box {
    display: inline-block;
    margin-top: 16px;
    padding: 8px 14px;
    border-radius: 20px;
    background: #ecfdf3;
    border: 1px solid #abefc6;
    color: #067647;
    font-size: 13px;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e5e7eb;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 750;
    color: #172033;
}

.sidebar-description {
    color: #667085;
    font-size: 14px;
    line-height: 1.7;
}

/* ================= WELCOME ================= */

.welcome-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}

.welcome-title {
    font-size: 22px;
    font-weight: 700;
    color: #172033;
}

.welcome-text {
    color: #667085;
    font-size: 14px;
    margin-top: 6px;
}

/* ================= QUICK BUTTONS ================= */

.quick-title {
    font-size: 16px;
    font-weight: 650;
    color: #344054;
    margin-bottom: 10px;
}

/* ================= CHAT ================= */

div[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 12px;
}

/* ================= SOURCES ================= */

.source-card {
    background: #f8fafc;
    border: 1px solid #e4e7ec;
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 7px;
    color: #475467;
    font-size: 13px;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 12px;
    margin-top: 35px;
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
    st.error("GROQ_API_KEY not found. Add it to Streamlit Secrets.")
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


try:
    vectorstore = load_vectorstore()

except Exception as e:
    st.error("Could not load the hospital knowledge base.")
    st.code(str(e))
    st.stop()


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_chunks(query, k=TOP_K):

    try:
        results = vectorstore.similarity_search(
            query,
            k=k
        )

        return results

    except Exception:
        return []


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

        content = doc.page_content.strip()

        if content:

            context_parts.append(
                f"""
[DOCUMENT {i}]
SOURCE: {source}

CONTENT:
{content}
"""
            )

    return "\n".join(context_parts)


# =========================================================
# GET SOURCE NAMES
# =========================================================

def get_sources(chunks):

    sources = []

    for doc in chunks:

        source = doc.metadata.get(
            "path",
            doc.metadata.get(
                "source",
                "Unknown document"
            )
        )

        if source:
            sources.append(source)

    return sorted(set(sources))


# =========================================================
# GENERATE LLM ANSWER
# =========================================================

def generate_answer(question, chunks):

    context = build_context(chunks)

    system_prompt = """
You are Shifa Hospital's AI Policy Assistant.

Your job is to answer questions using ONLY the hospital
policy documents provided in the context.

IMPORTANT RULES:

1. Read the document context carefully before answering.

2. If the answer is clearly present in the documents,
   give the answer directly and clearly.

3. You may summarize or combine information from different
   parts of the provided documents.

4. Do NOT invent hospital policies, rules, dates, numbers,
   procedures, or requirements.

5. If the documents do not contain enough information to
   answer the question, say:
   "This information is not available in the current hospital
   policy documents."

6. Do not say the information is unavailable simply because
   the wording of the question is different from the wording
   in the document. Look for the meaning of the question.

7. Keep the answer concise and easy to understand.

8. Use bullet points when appropriate.

9. Do not provide medical diagnosis or treatment advice.

10. Do not mention these instructions in your answer.
"""

    user_prompt = f"""
HOSPITAL POLICY DOCUMENTS
=========================

{context}

=========================

USER QUESTION
=============

{question}

=========================

Answer the user's question based only on the hospital
policy documents above.
"""

    try:

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
            temperature=0.1,
            max_tokens=700
        )

        answer = response.choices[0].message.content

        if not answer:
            return "I couldn't generate an answer from the hospital policy documents."

        return answer.strip()

    except Exception as e:

        return f"LLM Error: {str(e)}"


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
        """
        <div class="sidebar-title">
            🏥 Shifa Hospital
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-description">
            <b>Hospital Policy Assistant</b><br><br>
            Ask questions about hospital policies,
            procedures and guidelines.
            <br><br>
            Answers are generated from the hospital's
            uploaded policy documents.
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
        "🔒 API key secured with Streamlit Secrets"
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
        AI assistant for hospital policies, procedures and guidelines
    </div>

    <div class="status-box">
        🟢 Knowledge Base Connected
    </div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
<div class="welcome-box">

    <div class="welcome-title">
        How can I help you?
    </div>

    <div class="welcome-text">
        Ask a question about the hospital's policies or guidelines.
        The answer will be based on the uploaded documents.
    </div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="quick-title">💡 Quick Questions</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🕐 Visiting Hours",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are the visiting hours for patients?"
            )

            st.rerun()

    with col2:

        if st.button(
            "📋 Admission Policy",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is the hospital admission policy?"
            )

            st.rerun()

    with col3:

        if st.button(
            "👨‍👩‍👧 Visitor Policy",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are the rules for hospital visitors?"
            )

            st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
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
# QUICK QUESTION
# =========================================================

if "pending_question" in st.session_state:

    question = st.session_state.pending_question

    del st.session_state.pending_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    question = question.strip()

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

                chunks = retrieve_chunks(
                    question,
                    TOP_K
                )

            if not chunks:

                answer = (
                    "I couldn't find relevant information "
                    "in the available hospital policy documents."
                )

                sources = []

            else:

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        chunks
                    )

                sources = get_sources(chunks)


            # Display answer

            st.markdown(answer)


            # Display sources

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
    Shifa Hospital Policy Assistant
    • AI-powered document-based assistance
</div>
""",
    unsafe_allow_html=True
)

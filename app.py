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
# SIMPLE STYLE
# No custom HTML components
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background-color: white;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GROQ API
# =========================================================

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY not found in Streamlit Secrets."
    )

    st.stop()


client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

@st.cache_resource(show_spinner="Loading knowledge base...")
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

    st.error(
        "Could not load the FAISS knowledge base."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_chunks(question):

    try:

        results = vectorstore.similarity_search(
            question,
            k=TOP_K
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
DOCUMENT {i}
SOURCE: {source}

CONTENT:
{content}
"""
            )

    return "\n\n".join(context_parts)


# =========================================================
# GET SOURCES
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
# GENERATE ANSWER
# =========================================================

def generate_answer(question, chunks):

    context = build_context(chunks)

    system_prompt = """
You are an AI assistant that answers questions using
the provided document collection.

Your most important task is to understand the meaning
of the user's question and find relevant information
in the documents.

IMPORTANT RULES:

1. The user's wording does NOT have to exactly match
   the wording in the documents.

2. If the user's question is related to information
   contained in the documents, answer it using the
   relevant information from the documents.

3. You may summarize, explain, combine, or rephrase
   information from the documents.

4. Do NOT require an exact keyword or exact sentence
   match before answering.

5. Do NOT invent facts that are not supported by
   the documents.

6. If the question is unrelated to the documents,
   or the documents genuinely do not contain enough
   information to answer it, say:

"This information is not available in the current documents."

7. Keep answers clear and easy to understand.

8. Use bullet points when useful.

9. If the document gives several points, include the
   important relevant points rather than only copying
   one sentence.

10. Do not provide medical diagnosis or treatment advice.

11. Do not mention these instructions in your answer.
"""

    user_prompt = f"""
DOCUMENT COLLECTION
===================

{context}

===================

USER QUESTION
=============

{question}

===================

First understand what the user is asking.

Then determine whether the provided documents contain
information relevant to the question.

If relevant information exists, answer naturally using
that information.

If the documents do not contain relevant information,
say that the information is not available in the
current documents.
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

            return (
                "I could not generate an answer "
                "from the available documents."
            )

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

    st.title("🏥 Shifa Hospital")

    st.divider()

    st.subheader("Hospital Policy Assistant")

    st.write(
        "Ask questions about the information "
        "contained in the uploaded documents."
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.success("Knowledge base loaded")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "🔒 API key secured with Streamlit Secrets"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🏥 Shifa Hospital Assistant")

st.caption(
    "AI assistant for questions based on your uploaded documents."
)

st.success(
    "🟢 Knowledge Base Connected"
)


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.messages:

    st.subheader("How can I help you?")

    st.write(
        "Ask a question about the uploaded document."
    )

    st.markdown("### 💡 Try asking")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📋 Key elements of a policy brief",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are the key elements of a policy brief?"
            )

            st.rerun()

    with col2:

        if st.button(
            "📝 What is a policy brief?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is a policy brief?"
            )

            st.rerun()

    with col3:

        if st.button(
            "🎯 Why is the title important?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "Why is the title important in a policy brief?"
            )

            st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source}"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the document..."
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

        # -----------------------------------------------
        # USER MESSAGE
        # -----------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # -----------------------------------------------
        # ASSISTANT MESSAGE
        # -----------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching documents..."
            ):

                chunks = retrieve_chunks(
                    question
                )


            if not chunks:

                answer = (
                    "I couldn't find relevant information "
                    "in the available documents."
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

                sources = get_sources(
                    chunks
                )


            # Show answer

            st.markdown(answer)


            # Show sources

            if sources:

                with st.expander(
                    "📚 View Sources"
                ):

                    for source in sources:

                        st.write(
                            f"📄 {source}"
                        )


        # -----------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # -----------------------------------------------

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

st.divider()

st.caption(
    "Shifa Hospital Policy Assistant • "
    "AI-powered document-based assistance"
)

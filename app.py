
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Book AI Chatbot",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #101323, #171b30, #20213b);
    color: white;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label, span {
    color: white;
}

[data-testid="stChatMessage"] {
    background: #ffffff;
    color: #111827;
    border: 1px solid rgba(17,24,39,0.12);
    border-radius: 15px;
    padding: 15px;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    color: #111827 !important;
}

[data-testid="stChatInput"] {
    border-radius: 15px;
}

.st-key-chat_prompt textarea {
    color: #111827 !important;
    -webkit-text-fill-color: #111827;
    caret-color: #111827;
    background-color: #ffffff !important;
}

.st-key-chat_prompt textarea::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280;
}

section[data-testid="stSidebar"] {
    background: #14182a;
}

.stButton button {
    background: #5965d8;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
}

.stButton button:hover {
    background: #737ff0;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# API KEY VALIDATION
# --------------------------------------------------

if not GROQ_API_KEY or not GOOGLE_API_KEY:
    st.error(
        "API keys are missing. Please configure GROQ_API_KEY "
        "and GOOGLE_API_KEY in your .env file."
    )
    st.stop()


# --------------------------------------------------
# INITIALIZE EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def get_embedding_model(api_key):

    return init_embeddings(
        model="gemini-embedding-001",
        provider="google_genai",
        api_key=api_key
    )


embedding_model = get_embedding_model(GOOGLE_API_KEY)


# --------------------------------------------------
# INITIALIZE CHAT MODEL
# --------------------------------------------------

@st.cache_resource
def get_chat_model(api_key):

    return init_chat_model(
        model="openai/gpt-oss-20b",
        model_provider="groq",
        api_key=api_key,
        temperature=0.7
    )


chat_model = get_chat_model(GROQ_API_KEY)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_book_hash" not in st.session_state:
    st.session_state.active_book_hash = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None


# --------------------------------------------------
# PDF PROCESSING AND VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def build_retriever(pdf_bytes, book_hash, _embedding_model):

    # Create temporary PDF file

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:

        # Load PDF

        documents = PyPDFLoader(temp_path).load()

    finally:

        Path(temp_path).unlink(missing_ok=True)

    if not documents:
        raise ValueError("No readable pages were found in this PDF.")

    # Split documents into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    # Database directory for this particular book

    database_path = Path("Chroma_db") / book_hash
    database_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name="book_collection",
        persist_directory=str(database_path),
        embedding_function=_embedding_model
    )

    # Generate unique IDs for chunks

    chunk_ids = [
        f"{index}-{hashlib.sha256(
            chunk.page_content.encode('utf-8')
        ).hexdigest()}"
        for index, chunk in enumerate(chunks)
    ]

    # Check existing embeddings

    existing_ids = set(
        vectorstore.get(include=[])["ids"]
    )

    # Remove outdated collection if necessary

    if not existing_ids.issubset(set(chunk_ids)):

        vectorstore.delete_collection()

        vectorstore = Chroma(
            collection_name="book_collection",
            persist_directory=str(database_path),
            embedding_function=_embedding_model
        )

        existing_ids.clear()

    # Add only missing chunks

    missing_chunks = [
        (chunk, chunk_id)
        for chunk, chunk_id in zip(chunks, chunk_ids)
        if chunk_id not in existing_ids
    ]

    if missing_chunks:

        vectorstore.add_documents(
            documents=[
                chunk for chunk, _ in missing_chunks
            ],
            ids=[
                chunk_id for _, chunk_id in missing_chunks
            ]
        )

    # Create MMR retriever

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return retriever, len(chunks)


# --------------------------------------------------
# CHAT PROMPT TEMPLATE
# --------------------------------------------------

prompt_template = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful AI Book Assistant.

        Your job is to answer questions using the uploaded book.

        Follow these rules:

        1. Use ONLY the provided document context for factual
           information about the book.

        2. You can use conversation history to understand
           follow-up questions and references.

        3. Never invent information that is not present
           in the document.

        4. If the answer is not available in the document,
           say exactly:

           "I could not find the answer in the document."

        5. Give clear, structured and easy-to-understand answers.

        6. Avoid repeating unnecessary information.

        Conversation History:
        {chat_history}

        Document Context:
        {context}
        """
    ),

    (
        "human",
        """
        Question: {question}
        """
    )

])


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("Book Settings")

    st.write("Upload a PDF book to start chatting.")

    uploaded_file = st.file_uploader(
        "Upload your book",
        type=["pdf"]
    )

    st.divider()

    if st.session_state.book_name:

        st.success(
            f"Active Book: {st.session_state.book_name}"
        )

    if st.button("Clear Chat History"):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Powered by Groq, Gemini and ChromaDB")


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

st.title("Book AI Chatbot", text_alignment="center")
st.markdown(
    "Upload your book and ask questions. "
    "Your AI assistant will help you understand it.",
    text_alignment="center",
)


# --------------------------------------------------
# HANDLE PDF UPLOAD
# --------------------------------------------------

if uploaded_file is None:

    st.info("Please upload a PDF book using the sidebar.")

    st.stop()


pdf_bytes = uploaded_file.getvalue()

book_hash = hashlib.sha256(pdf_bytes).hexdigest()

# Detect a newly uploaded book

if st.session_state.active_book_hash != book_hash:

    st.session_state.messages = []
    st.session_state.retriever = None

    with st.spinner("Processing your book..."):

        try:

            retriever, total_chunks = build_retriever(
                pdf_bytes,
                book_hash,
                embedding_model
            )

            st.session_state.retriever = retriever
            st.session_state.active_book_hash = book_hash
            st.session_state.book_name = uploaded_file.name

            st.success(
                f"Book processed successfully! "
                f"{total_chunks} chunks created."
            )

        except Exception as e:

            st.error(f"Error processing PDF: {e}")
            st.stop()


# --------------------------------------------------
# DISPLAY BOOK INFORMATION
# --------------------------------------------------

st.markdown(
    f"**Currently reading:** {st.session_state.book_name}"
)

st.divider()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_input = st.chat_input(
    "Ask something about your book...",
    key="chat_prompt",
)


# --------------------------------------------------
# GENERATE RESPONSE
# --------------------------------------------------

if user_input:

    # Display user message

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                # Build conversation memory from previous messages.
                # Keep the latest 10 messages.

                previous_messages = st.session_state.messages[:-1]

                history = previous_messages[-10:]

                chat_history = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: "
                    f"{msg['content']}"
                    for msg in history
                ])

                # Include recent conversation in retrieval query
                # to improve follow-up question retrieval.

                retrieval_query = "\n".join([
                    f"{msg['content']}"
                    for msg in history[-4:]
                    if msg["role"] == "user"
                ])

                retrieval_query += "\n" + user_input

                # Retrieve relevant document chunks

                retrieved_docs = st.session_state.retriever.invoke(
                    retrieval_query
                )

                context = "\n\n".join([
                    doc.page_content
                    for doc in retrieved_docs
                ])

                # Create final prompt

                final_prompt = prompt_template.invoke({
                    "context": context,
                    "question": user_input,
                    "chat_history": chat_history
                })

                # Generate answer

                response = chat_model.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

        # Save assistant response to memory

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:

        error_message = f"An error occurred: {e}"

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })

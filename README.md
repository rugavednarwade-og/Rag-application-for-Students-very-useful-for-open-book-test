# Book AI Chatbot

A Streamlit-based AI assistant that lets you upload a PDF book and ask
questions about its contents. It uses retrieval-augmented generation
(RAG) to find relevant passages from the uploaded PDF and uses a
language model to generate answers grounded in those passages.

## Features

-   Upload a PDF book through the sidebar.
-   Extract text from the PDF and split it into smaller chunks.
-   Create and store vector embeddings with ChromaDB.
-   Retrieve relevant passages using Maximal Marginal Relevance (MMR).
-   Ask questions about the uploaded book through a chat interface.
-   Use recent conversation history to help interpret follow-up
    questions.
-   Clear the chat history without re-uploading the book.
-   Cache the embedding model, chat model, and retriever resources to
    reduce repeated initialization and processing.

## Technologies Used

-   **Python** --- application language
-   **Streamlit** --- web interface
-   **LangChain** --- model, prompt, document, and retrieval
    orchestration
-   **Groq** --- chat model provider (`openai/gpt-oss-20b`)
-   **Google Gemini** --- text embeddings (`gemini-embedding-001`)
-   **ChromaDB** --- vector database
-   **PyPDF** --- PDF text extraction
-   **python-dotenv** --- environment variable loading

## Project Setup

### 1. Clone or download the project

Place the Python application file in your project folder. The
instructions below assume it is named `app.py`; replace that name if
your file is different.

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

On Windows:

``` bash
.venv\Scripts\activate
```

On macOS or Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install streamlit python-dotenv langchain langchain-community langchain-text-splitters langchain-groq langchain-google-genai chromadb pypdf
```

If your installed LangChain versions differ, check the relevant provider
and integration package documentation for compatible versions.

### 4. Configure API keys

Create a file named `.env` in the project folder:

``` env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Get API keys from the Groq and Google AI developer platforms. Keep the
`.env` file private and do not commit it to a public repository.

The application checks that both keys are present before it starts.

### 5. Run the application

``` bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in
your browser.

## How to Use

1.  Start the application.
2.  In the sidebar, choose **Upload your book** and select a PDF file.
3.  Wait while the application extracts the text, splits it into chunks,
    creates embeddings, and prepares the retriever.
4.  Type a question in the chat input at the bottom of the page.
5.  Continue asking follow-up questions about the same book.
6.  Select **Clear Chat History** in the sidebar to clear the
    conversation.

Uploading a different PDF resets the current conversation and processes
the newly selected book.

## How It Works

### 1. PDF loading

The uploaded PDF is temporarily saved so `PyPDFLoader` can read its
pages. The temporary file is deleted after loading.

### 2. Text splitting

The extracted document is split with `RecursiveCharacterTextSplitter`
using:

-   Chunk size: `1000` characters
-   Chunk overlap: `100` characters

### 3. Embeddings and vector storage

The application uses Google's `gemini-embedding-001` embedding model to
convert document chunks into vectors. ChromaDB stores those vectors in a
book-specific folder under `Chroma_db/`.

A SHA-256 hash of the PDF content identifies the book. Chunk IDs are
also generated from the chunk index and content hash so existing chunks
can be reused when possible.

### 4. Retrieval

When a question is asked, the retriever searches for relevant chunks
using MMR (Maximal Marginal Relevance). It requests up to 3 results,
considering up to 10 candidates.

### 5. Answer generation

The retrieved passages, the question, and recent conversation history
are inserted into a `ChatPromptTemplate`. The Groq chat model
(`openai/gpt-oss-20b`) then generates the answer.

The prompt instructs the assistant to use only the supplied document
context for factual information about the book. If the answer is not
present, it is instructed to reply:

> I could not find the answer in the document.

## Project Data

The application creates a `Chroma_db/` directory to store vector
database data. This directory may become large as more books are
processed.

The current chat history and active-book information are stored in
Streamlit session state. The code does not provide a separate
chat-history export or a user account system.

## Important Notes and Limitations

-   The application currently accepts PDF files only.
-   Answers depend on the text that can be extracted from the PDF.
    Scanned PDFs or image-only pages may need OCR before their text can
    be searched.
-   The assistant is instructed to answer book-related factual questions
    from retrieved document context. Retrieval can miss relevant
    passages, so answers may occasionally be incomplete.
-   API access and usage limits depend on the Groq and Google services
    and the API plans in use.
-   The `.env` file contains secrets. Add `.env` and other private files
    to `.gitignore`.
-   The code uses local ChromaDB persistence in `Chroma_db/`. Keep this
    directory if you want to retain the stored vectors between runs.

## Suggested `.gitignore`

``` gitignore
.env
.venv/
__pycache__/
*.pyc
Chroma_db/
.streamlit/secrets.toml
```

## License

No license is specified in the provided program. Add a license file if
you intend to distribute or publish the project.

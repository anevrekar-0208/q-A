from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI

# === Embeddings using HuggingFace (FREE + no API key required) ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# === Language Model using OpenRouter ===
llm = ChatOpenAI(
    model_name="mistralai/mistral-small-3.2-24b-instruct:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# === Vectorstore settings ===
VECTORSTORE_DIR = "vectorstore"
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# === Load documents from disk ===
def load_file(file_path: str) -> list[Document]:
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf8")
    return loader.load()

# === Add uploaded file to vectorstore ===
def add_documents_to_vectorstore(file) -> None:
    suffix = ""
    filename = "uploaded_file"
    if hasattr(file, "name") and file.name:
        suffix = os.path.splitext(file.name)[1].lower()
        filename = file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    docs = load_file(tmp_path)
    texts = text_splitter.split_documents(docs)

    # ✅ Store original filename in metadata (for UI display)
    for doc in texts:
        doc.metadata["source"] = filename

    # ✅ Save to Chroma vectorstore
    if os.path.exists(VECTORSTORE_DIR):
        vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
    else:
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=VECTORSTORE_DIR)

    vectorstore.persist()

# === Load existing vectorstore from disk ===
def load_existing_vectorstore():
    if not os.path.exists(VECTORSTORE_DIR):
        return None
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

# === Ask a question using RAG ===
def ask_with_context(query: str) -> str:
    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        return "⚠️ No documents found. Please upload a document first."

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

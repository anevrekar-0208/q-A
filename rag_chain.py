import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# ✅ API keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ✅ Model configuration
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
LLM_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"
VECTORSTORE_DIR = "vectorstore"
DOCS_DIR = "docs"

# Set HuggingFace API token as environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

# ✅ Use recommended HuggingFaceEmbeddings class
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_and_split_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")

    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return [doc for doc in split_docs if doc.page_content.strip()]

def load_docs_from_folder():
    all_docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if path.endswith(".pdf") or path.endswith(".txt"):
            try:
                docs = load_and_split_file(path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return all_docs

def add_documents_to_vectorstore(docs, persist_directory=VECTORSTORE_DIR):
    embeddings = get_embeddings()
    texts = [doc.page_content for doc in docs]

    print(f"➡️ Getting embeddings for {len(texts)} chunks...")
    try:
        embedding_vectors = embeddings.embed_documents(texts)
        print(f"✅ Got {len(embedding_vectors)} embeddings.")
        if not embedding_vectors or len(embedding_vectors) != len(texts):
            raise ValueError("Mismatch in embedding count or failed embeddings.")
    except Exception as e:
        print("❌ Embedding failed:", e)
        raise

    print("📦 Building Chroma vectorstore...")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    #vectorstore.persist()
    print("✅ Vectorstore created and saved.")
    return vectorstore

def load_existing_vectorstore(persist_directory=VECTORSTORE_DIR):
    if not os.path.exists(persist_directory):
        return None
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

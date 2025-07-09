import streamlit as st
from rag_chain import add_documents_to_vectorstore, ask_with_context, load_existing_vectorstore
import os
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")


st.set_page_config(page_title="📄 Document Q&A Chatbot")

st.title("📄 Document Q&A Chatbot")
st.markdown("Upload PDF or TXT files to add to the knowledge base, then ask questions!")

# === Load existing vectorstore once ===
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = load_existing_vectorstore() is not None
    st.session_state.docs_added = False

# === File upload section ===
uploaded_files = st.file_uploader(
    "Upload a PDF or TXT file", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        add_documents_to_vectorstore(file)
    st.session_state.vectorstore_loaded = True
    st.session_state.docs_added = True
    st.success("✅ Documents added to knowledge base!")

# === Show filenames from metadata ===
if st.session_state.vectorstore_loaded:
    vectorstore = load_existing_vectorstore()
    try:
        metadatas = vectorstore.get()["metadatas"]
        unique_sources = sorted(set(d.get("source", "Unknown") for d in metadatas))

        st.markdown("### 🗂️ Preloaded Documents:")
        for src in unique_sources:
            st.markdown(f"- `{src}`")
    except:
        st.warning("⚠️ Could not load document names.")

# === Ask questions section ===
if st.session_state.vectorstore_loaded:
    question = st.text_input("🔍 Ask a question about the documents:")
    if question:
        with st.spinner("Thinking..."):
            try:
                answer = ask_with_context(question)
                st.markdown("**🧠 Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error while generating answer:\n\n{e}")
else:
    st.info("ℹ️ Upload a document to start or make sure `vectorstore/` exists.")

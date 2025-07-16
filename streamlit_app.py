import os
import streamlit as st
from rag_chain import (
    load_and_split_file,
    add_documents_to_vectorstore,
    load_existing_vectorstore,
    create_qa_chain,
    VECTORSTORE_DIR,
    DOCS_DIR,
)

st.title("📄 Document Q&A Chatbot")
st.write("Upload PDF or TXT files and ask questions about documents in the `docs/` folder!")

# Show files in docs folder
files_in_docs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith((".pdf", ".txt"))]

if files_in_docs:
    st.write("### Documents in `docs/` folder:")
    for f in files_in_docs:
        st.write(f"- {f}")
else:
    st.info("No documents found in the `docs/` folder.")

# Upload files
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"✅ Saved {len(uploaded_files)} file(s) to `docs/` folder.")

    if st.button("🔄 Reindex documents to update vectorstore"):
        with st.spinner("Reindexing documents..."):
            docs = []
            for filename in os.listdir(DOCS_DIR):
                if filename.lower().endswith((".pdf", ".txt")):
                    try:
                        docs.extend(load_and_split_file(os.path.join(DOCS_DIR, filename)))
                    except Exception as e:
                        st.error(f"Failed loading {filename}: {e}")

            if docs:
                try:
                    vectorstore = add_documents_to_vectorstore(docs, VECTORSTORE_DIR)
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ Vectorstore updated with new documents!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to create vectorstore: {e}")
            else:
                st.warning("No documents loaded for indexing.")

# Load vectorstore once on app start (or from session state)
if "vectorstore" not in st.session_state:
    vectorstore = load_existing_vectorstore(VECTORSTORE_DIR)
    if vectorstore:
        st.session_state.vectorstore = vectorstore

# QA interface always visible
if "vectorstore" in st.session_state:
    qa_chain = create_qa_chain(st.session_state.vectorstore)
    st.write("### Ask a question about the documents:")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Thinking..."):
            try:
                answer = qa_chain.run(query)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.write("### Ask a question about the documents:")
    st.info("⚠️ No indexed documents found. Please upload files and click reindex to enable Q&A.")
    st.text_input("Your question:", disabled=True)

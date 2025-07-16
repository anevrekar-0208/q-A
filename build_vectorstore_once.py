from rag_chain import load_docs_from_folder, add_documents_to_vectorstore

docs = load_docs_from_folder()
if not docs:
    print("No documents found in the docs/ folder.")
else:
    add_documents_to_vectorstore(docs)
    print("✅ Vectorstore built and saved.")
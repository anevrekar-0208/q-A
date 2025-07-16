import os
from rag_chain import (
    load_existing_vectorstore,
    add_documents_to_vectorstore,
    load_and_split_file,
    create_qa_chain,
    VECTORSTORE_DIR,
    DOCS_DIR,
)

def rebuild_vectorstore_from_folder():
    if os.path.exists(VECTORSTORE_DIR):
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if path.endswith(".pdf") or path.endswith(".txt"):
            docs.extend(load_and_split_file(path))
    if not docs:
        print("No documents found in the docs folder.")
        return None
    return add_documents_to_vectorstore(docs)

def main():
    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        print("No existing vectorstore found. Building from docs folder...")
        vectorstore = rebuild_vectorstore_from_folder()
        if vectorstore is None:
            print("Failed to build vectorstore. Exiting.")
            return

    qa_chain = create_qa_chain(vectorstore)

    print("You can now ask questions about your documents. Type 'exit' to quit.")
    while True:
        query = input("Ask your question: ").strip()
        if query.lower() == "exit":
            break
        answer = qa_chain.run(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

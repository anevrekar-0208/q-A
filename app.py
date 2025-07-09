import os
from rag_chain import add_documents_to_vectorstore

def build_vectorstore(doc_folder="docs"):
    for filename in os.listdir(doc_folder):
        if filename.lower().endswith((".pdf", ".txt")):
            filepath = os.path.join(doc_folder, filename)
            print(f"Processing {filename}...")
            with open(filepath, "rb") as f:
                add_documents_to_vectorstore(f)
    print("✅ Vectorstore built and saved.")

if __name__ == "__main__":
    build_vectorstore()

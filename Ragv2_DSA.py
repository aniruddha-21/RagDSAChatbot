from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import shutil
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


BASE_DIR = Path(__file__).parent
pdf_path = BASE_DIR / "DS_Complete.pdf"

CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "DS_Complete"

'''Loading the pdf'''
def load_pdf():
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )
    pdf_loader = PyPDFLoader(str(pdf_path))
    rawdoc = pdf_loader.load()
    print(f"Loaded {len(rawdoc)} pages from: {pdf_path.name}")
    print("\nFirst page preview:\n")
    print(rawdoc[0].page_content[:500])
    return rawdoc


'''Splitting the docs into overlapping chunks'''

def splitting(rawdoc):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap= 200)
    
    chunks = splitter.split_documents(rawdoc)

    print(f"\nCreated {len(chunks)} chunks.")
    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content[:500])
    print(f"\nMetadata: {chunks[0].metadata}")

    return chunks

'''Embedding the data'''
def embedding_vector(chunks):

    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print("\nRemoved the previous ChromaDB database.")

    embeddings = OllamaEmbeddings(model="all-minilm")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding= embeddings,
        collection_name= COLLECTION_NAME,
        persist_directory= str(CHROMA_PATH)
    )

    print(f"\nSaved {vector_store._collection.count()} chunks to ChromaDB.")
    print(f"Database folder: {CHROMA_PATH}")
    return vector_store

"""Print the most relevant chunks for a test question."""

def test_retrieval(vector_store, question):
    results = vector_store.similarity_search(question, k=3)

    print(f"\nQuestion: {question}")
    print("\nTop retrieved chunks:")

    for number, document in enumerate(results, start=1):
        page = document.metadata.get("page", "Unknown")

        print(f"\n--- Result {number} | PDF page {page + 1} ---")
        print(document.page_content[:500])



if __name__ == "__main__":
    rawdoc = load_pdf()
    chunks = splitting(rawdoc)
    vector_store = embedding_vector(chunks)
    
    test_retrieval(
        vector_store,
        "What is an Linkedlist?",
    )
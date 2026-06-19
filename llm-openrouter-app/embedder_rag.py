import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

embed_model_id = "intfloat/multilingual-e5-base"  # model wielojęzyczny, radzi sobie z polskim
model_kwargs = {"device": "cpu", "trust_remote_code": True}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs=model_kwargs,
    )

def chunk_documents(documents):
    """Dzieli każdy dokument na nakładające się fragmenty o stałej długości w znakach."""
    chunks = []
    for doc in documents:
        text = doc["text"].replace("\n", " ").strip()
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            fragment = text[start:end].strip()
            if fragment:
                chunks.append({
                    "filename": doc["filename"],
                    "text": fragment,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=4):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

def create_index(documents):
    embeddings = get_embeddings()
    chunks = chunk_documents(documents)  # podział na fragmenty przed embedowaniem
    texts = [chunk["text"] for chunk in chunks]
    metadata = chunks

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=4):
    embeddings = get_embeddings()
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    results = faiss_index.similarity_search(query_embedding, k)
    return results

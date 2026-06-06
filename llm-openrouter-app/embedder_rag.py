import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

embed_model_id = "intfloat/multilingual-e5-base"  # model wielojęzyczny, radzi sobie z polskim
model_kwargs = {"device": "cpu", "trust_remote_code": True}

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs=model_kwargs,
    )

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

def create_index(documents):
    embeddings = get_embeddings()
    texts = [doc["text"] for doc in documents]
    metadata = documents

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = get_embeddings()
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    results = faiss_index.similarity_search(query_embedding, k)
    return results

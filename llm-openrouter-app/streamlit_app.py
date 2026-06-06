import streamlit as st
from chat_openrouter import ChatOpenRouter
import os
from langchain_core.prompts import ChatPromptTemplate
import shutil
from docloader import load_documents_from_folder
from embedder_rag import create_index, retrieve_docs

st.set_page_config(layout="wide", page_title="Rag chatbot app")
st.title("Rag chatbot app")

UPLOAD_FOLDER = "data/uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

template = """
Jesteś asystentem do odpowiadania na pytania zgodnie z Twoją bazą wiedzy. Odpowiadaj w maksymlanie 3 zdaniach
Question: {question} 
Context: {context} 
Answer:
"""

selected_model = "gemini-2.5-flash" # nazwa modelu
model = ChatOpenRouter(model_name=selected_model)

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template) # prompt template'owy
    chain = prompt | model # chain
    return chain.invoke({"question": question, "context": context})

if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "clear_files" not in st.session_state:
    st.session_state.clear_files = False
if "retrieve_files" not in st.session_state:
    st.session_state.retrieve_files = False
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="file_uploader")

if st.sidebar.button("Clear files"):
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    st.session_state.clear_files = True
    st.session_state.query = ""
    st.session_state.answer = ""
    st.sidebar.success("Uploaded files cleared!")

if st.session_state.clear_files:
    uploaded_files = None
    st.session_state.clear_files = False
    st.session_state.retrieve_files = False

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.write("Files uploaded successfully!")
    documents = load_documents_from_folder(UPLOAD_FOLDER)
    st.session_state.faiss_index = create_index(documents) # dodanie stanu bazy do sesji
    st.write("Files retrieved successfully!")
    st.session_state.retrieve_files = True

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Jak mogę pomóc?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    if st.session_state.retrieve_files:
        related_documents = retrieve_docs(question, st.session_state.faiss_index) # wyszukanie najbardziej pasujących do query dokumentów
        answer = answer_question(question, related_documents, model) # wywołanie odpowiedzi na pytanie z dodanymi do kontekstu dokumentami
    else:
        answer = answer_question(question, [], model)
    st.session_state.messages.append({"role": "assistant", "content": answer.content})
    st.chat_message("assistant").write(answer.content)

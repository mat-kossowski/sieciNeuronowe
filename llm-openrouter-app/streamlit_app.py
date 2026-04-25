import streamlit as st
from openai import OpenAI
from pdf_utils import extract_text_from_pdfs

st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "google/gemma-3-1b-it:free"

with st.sidebar:
    st.header("📄 Dokumenty PDF")
    uploaded_files = st.file_uploader(
        "Wgraj pliki PDF",
        type="pdf",
        accept_multiple_files=True
    )
    pdf_texts = []
    if uploaded_files:
        pdf_texts = extract_text_from_pdfs(uploaded_files)
        st.success(f"Wczytano {len(pdf_texts)} plik(ów)")
    if pdf_texts:
    for doc in pdf_texts:
        st.subheader(doc["name"])
        st.text(doc["text"])

st.title("OpenRouter chatbot app")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

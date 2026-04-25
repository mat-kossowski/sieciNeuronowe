import fitz

def extract_text_from_pdfs(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        texts.append({"name": uploaded_file.name, "text": text})
    return texts

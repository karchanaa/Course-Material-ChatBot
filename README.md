# Course-Material-ChatBot
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# -------------------------------
# Load model
# -------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

index = None
chunks = None

# -------------------------------
# Load & CLEAN PDF
# -------------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    # 🔥 Remove unwanted content
    text = re.sub(r"Prof\..*Sani.*", "", text)
    text = re.sub(r"Sudeshna\s*Sani.*", "", text)
    text = re.sub(r"Figure\s*\d+.*", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text

# -------------------------------
# Sentence-based splitting
# -------------------------------
def split_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks

# -------------------------------
# Vector store
# -------------------------------
def create_vector_store(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index

# -------------------------------
# Retrieve TOP 5 chunks
# -------------------------------
def retrieve(query):
    global index, chunks

    q = embed_model.encode([query])
    q = np.array(q).astype("float32")

    D, I = index.search(q, 5)
    results = [chunks[i] for i in I[0]]

    return results

# -------------------------------
# 🔥 SMART SUMMARIZER (KEY PART)
# -------------------------------
def summarize(text_list):
    text = " ".join(text_list)

    sentences = re.split(r'(?<=[.!?]) +', text)

    # remove small / useless sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    # take important top sentences
    top_sentences = sentences[:6]

    return top_sentences

# -------------------------------
# 🎯 FORMAT FINAL ANSWER
# -------------------------------
def format_answer(query, sentences):
    answer = f"### 📌 Answer for: {query}\n\n"

    for s in sentences:
        answer += f"• {s}\n\n"

    return answer

# -------------------------------
# Answer function
# -------------------------------
def answer_question(query):
    global index

    if index is None:
        return "❌ Upload and process PDF first."

    results = retrieve(query)
    summary = summarize(results)

    return format_answer(query, summary)

# -------------------------------
# Process PDF
# -------------------------------
def process_pdf(file):
    global index, chunks

    if file is None:
        return "❌ Upload PDF"

    text = load_pdf(file)

    if len(text) == 0:
        return "❌ PDF unreadable"

    chunks = split_text(text)
    index = create_vector_store(chunks)

    return "✅ PDF processed successfully!"

# -------------------------------
# UI
# -------------------------------
with gr.Blocks() as app:
    gr.Markdown("🤖 Smart Course Chatbot")

    file = gr.File(label="Upload PDF")
    btn = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")

    q = gr.Textbox(label="Ask Question")
    ask = gr.Button("Get Answer")
    out = gr.Textbox(label="Answer")

    btn.click(process_pdf, inputs=file, outputs=status)
    ask.click(answer_question, inputs=q, outputs=out)

# -------------------------------
# Launch
# -------------------------------
app.launch(server_name="0.0.0.0", server_port=7860)


# imports
import gradio as gr
import zipfile
import os

# zip function
def create_zip():
    ...

# your chatbot code
...

# gradio UI
demo = gr.Interface(...)

# call zip (optional)
create_zip()

# launch (ALWAYS LAST)
demo.launch()

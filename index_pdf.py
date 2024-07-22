# index_pdf.py

from elasticsearch import Elasticsearch
import fitz  # PyMuPDF
import nltk

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=350):
    words = nltk.word_tokenize(text)
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def index_chunks(chunks):
    es = Elasticsearch()
    index_name = "pdf_chunks"

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name)

    for i, chunk in enumerate(chunks):
        es.index(index=index_name, id=i, body={"text": chunk})

if __name__ == '__main__':
    pdf_path = r'C:\Users\IT\Desktop\Computer_hub\Corpus.pdf'
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    index_chunks(chunks)
    print("PDF chunks indexed successfully.")

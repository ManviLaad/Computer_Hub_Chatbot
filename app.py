from flask import Flask, request, jsonify, render_template
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.data import find
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure stopwords are downloaded
def download_stopwords():
    try:
        find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords')

def preprocess(text):
    download_stopwords()
    try:
        stopwords_list = set(stopwords.words('english'))
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        stopwords_list = set()  # Fallback to an empty set

    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stopwords_list]
    return ' '.join(words)

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_query', methods=['POST'])
def send_query():
    data = request.get_json()
    if not data:
        return jsonify({'response': "Invalid input data. Please send a valid JSON with a 'query' key."}), 400
    
    user_query = data.get('query')
    if not user_query:
        return jsonify({'response': "Query not found in the request data."}), 400
    
    print(f"Received Query: {user_query}")

    pdf_path = r'C:\Users\IT\Desktop\Computer_hub\Corpus.pdf'
    context = extract_text_from_pdf(pdf_path)

    if not context:
        return jsonify({'response': "Error extracting text from PDF."}), 500

    chunks = split_text_into_chunks(context)

    clean_chunks = [preprocess(chunk) for chunk in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_chunks)
    query_vec = vectorizer.transform([preprocess(user_query)])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    relevant_chunk_indices = similarities.argsort()[0][-5:]

    answers = []
    for idx in relevant_chunk_indices:
        chunk = chunks[idx]
        response = answer_question(user_query, chunk)
        answers.append(response)

    combined_answer = " ".join(answers).strip()
    combined_answer = combined_answer.replace('[CLS]', '').replace('[SEP]', '').strip()

    if not combined_answer:
        combined_answer = "Sorry, I couldn't find an answer to your question. Please contact us directly."

    print(f"Combined Answer: {combined_answer}")
    return jsonify({'response': combined_answer})


if __name__ == "__main__":
    app.run(debug=False)

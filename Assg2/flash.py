# image_search_engine.py

import os
import json
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from flask import Flask, request, render_template, send_from_directory

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load image metadata
with open("image_metadata_blip8.json", "r", encoding="utf-8") as f:
    images = json.load(f)

# Preprocess and build corpus
corpus = {}
doc_freq = {}
doc_term_counts = {}
idf_values = {}


def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]


for img in images:
    img_id = img["local_path"]
    surrogate_text = f"{img.get('alt_text', '')} {img.get('figcaption', '')} {img.get('title', '')} {img.get('vision_caption', '')}"
    tokens = preprocess(surrogate_text)
    corpus[img_id] = tokens

    term_counts = {}
    for t in tokens:
        term_counts[t] = term_counts.get(t, 0) + 1
    doc_term_counts[img_id] = term_counts

    for t in set(tokens):
        doc_freq[t] = doc_freq.get(t, 0) + 1

# Compute IDF
num_docs = len(corpus)
idf_values = {term: math.log((num_docs + 1) / (df + 1)) + 1 for term, df in doc_freq.items()}

# Compute TF-IDF vectors
def compute_tf(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    for t in tf:
        tf[t] /= len(tokens)
    return tf


def compute_tfidf_vector(tokens):
    tf = compute_tf(tokens)
    return {t: tf[t] * idf_values.get(t, 0) for t in tf}


vectors = {doc_id: compute_tfidf_vector(tokens) for doc_id, tokens in corpus.items()}


# VSM Search

def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in set(vec1) & set(vec2))
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def search(query, top_k=10):
    query_tokens = preprocess(query)
    query_vector = compute_tfidf_vector(query_tokens)
    results = []
    for doc_id, vec in vectors.items():
        sim = cosine_similarity(query_vector, vec)
        if sim > 0:
            results.append((doc_id, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]


# Flask App Setup
app = Flask(__name__)

@app.route("/<path:filename>")
def serve_image(filename):
    return send_from_directory("", filename)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        raw_results = search(query)
        results = [(path.replace("\\", "/"), score) for path, score in raw_results]
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
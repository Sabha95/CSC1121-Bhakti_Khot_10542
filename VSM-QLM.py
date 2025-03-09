import math
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# File paths
DATASET_PATH = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\cran.all.1400.xml"
QUERY_PATH = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\cran.qry.xml"

# Output files
OUTPUT_VSM = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\search_results_vsm.txt"
OUTPUT_BOOLEAN = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\search_results_boolean.txt"
OUTPUT_QLM = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\search_results_qlm.txt"

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# **Step 1: Read & Preprocess Documents**
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Lowercasing & tokenization
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Inverted index for Boolean search
inverted_index = defaultdict(list)
doc_freq = defaultdict(int)
doc_length = defaultdict(int)
doc_term_counts = {}
unique_terms = {}
collection_term_counts = defaultdict(int)
collection_length = 0
corpus = {}
doc_texts = {}

tree = ET.parse(DATASET_PATH)
root = tree.getroot()

def build_inverted_index():
    """
    Builds an inverted index for Boolean search.

    This function iterates over each document in the corpus, preprocesses the text, and
    stores the preprocessed tokens in the `corpus` dictionary. It also updates the
    `unique_terms`, `doc_length`, `doc_term_counts`, `doc_freq`, `collection_term_counts`,
    and `collection_length` dictionaries accordingly.
    """
    global collection_length  # Ensure it's updated globally
    for doc in root.findall("doc"):
        doc_id = doc.find("docno").text.strip()
        text = (doc.find("title").text or "") + " " + (doc.find("text").text or "")
        doc_texts[doc_id] = text
        processed_tokens = preprocess_text(text)
        corpus[doc_id] = processed_tokens
        unique_terms = set(processed_tokens)
        doc_length[doc_id] = len(processed_tokens)
        
        term_counts = defaultdict(int)
        for term in processed_tokens:
            term_counts[term] += 1
            collection_term_counts[term] += 1
            collection_length += 1

        doc_term_counts[doc_id] = term_counts

        for term in unique_terms:
            # Store the document ID in the inverted index for Boolean search
            if doc_id not in inverted_index[term]:
                inverted_index[term].append(doc_id)
            doc_freq[term] += 1

build_inverted_index()


# **Step 2: Compute TF-IDF for VSM**
def compute_tf(doc_tokens):
    tf = defaultdict(float)
    for term in doc_tokens:
        tf[term] += 1
    doc_length = len(doc_tokens)
    for term in tf:
        tf[term] /= doc_length
    return tf

def compute_idf():
    num_docs = len(corpus)
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log((num_docs + 1) / (df + 1)) + 1  # Smoothed IDF
    return idf

idf_values = compute_idf()

def compute_tfidf_vector(doc_tokens):
    tf = compute_tf(doc_tokens)
    return {term: tf[term] * idf_values.get(term, 0) for term in tf}

vectors = {doc_id: compute_tfidf_vector(doc_tokens) for doc_id, doc_tokens in corpus.items()}

# **Step 3: Query Processing**
def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) & set(vec2))
    norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def vector_space_retrieval(query):
    query_tokens = preprocess_text(query)
    query_vec = compute_tfidf_vector(query_tokens)
    results = [(doc_id, cosine_similarity(query_vec, doc_vec)) for doc_id, doc_vec in vectors.items()]
    return sorted(results, key=lambda x: x[1], reverse=True)[:10]


def query_likelihood_retrieval(query, smoothing_param=0.1):
    """
    Computes the query likelihood retrieval model for the given query.
    
    Returns the top 100 documents with normalized positive scores.
    """
    query_tokens = preprocess_text(query)
    scores = {}
    min_score = float('inf')
    
    for doc_id, term_counts in doc_term_counts.items():
        doc_length = sum(term_counts.values())
        log_score = 0
        
        for term in query_tokens:
            term_count = term_counts.get(term, 0)
            collection_prob = collection_term_counts.get(term, 0) / (collection_length + 1e-10)
            smoothed_prob = (term_count + smoothing_param * collection_prob) / (doc_length + smoothing_param)
            
            if smoothed_prob > 0:
                log_score += math.log(smoothed_prob)
            else:
                log_score += math.log(1e-10)
        
        scores[doc_id] = log_score
        min_score = min(min_score, log_score)
    
    # Normalize scores to make them positive
    # This shifts all scores so the minimum score becomes 0
    for doc_id in scores:
        # Shift scores to make them positive and add 1 to ensure no zeros
        scores[doc_id] = abs(min_score) + scores[doc_id] + 1
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

query_tree = ET.parse(QUERY_PATH)
query_root = query_tree.getroot()
queries = {}
query_id = 1
for doc in query_root.findall("top"):
    # query_id = doc.find("num").text.strip()
    query_text = doc.find("title").text.strip()
    queries[query_id] = query_text
    query_id += 1

with open(OUTPUT_VSM, "w") as vsm_file, open(OUTPUT_BOOLEAN, "w") as boolean_file, open(OUTPUT_QLM, "w") as qlm_file:
    for query_id, query_text in queries.items():
        vsm_results = vector_space_retrieval(query_text)
       
        qlm_results = query_likelihood_retrieval(query_text)

        # Write VSM results
        for rank, (doc_id, score) in enumerate(vsm_results, start=1):
            if score > 0:
                score = 1
            else:
                score = 0
            vsm_file.write(f"{query_id} 0 {doc_id} {rank} {int(score)} VSM\n")
        
        # Write QLM results - now with positive scores
        for rank, (doc_id, score) in enumerate(qlm_results, start=1):
            if score > 0:
                score = 1
            else:
                score = 0
            qlm_file.write(f"{query_id} 0 {doc_id} {rank} {int(score)} QLM\n")
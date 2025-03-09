import math
import xml.etree.ElementTree as ET
import operator
from collections import defaultdict

# File Paths
DATASET_PATH = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\cran.all.1400.xml"
QUERY_PATH = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\cran.qry.xml"
OUTPUT_BM25 = "C:\\Users\\Bhakti\\Downloads\\Mechanics_Of_Search\\cranfield-trec-dataset\\search_results_bm25.txt"

# Data Structures
queriesdict = {}
inverted_index = defaultdict(dict)
doc_length = {}
dls = {}  # Document lengths
collection_term_counts = defaultdict(int)
collection_length = 0
N = 0  # Number of documents

# Load Cranfield Dataset
tree = ET.parse(DATASET_PATH)
root = tree.getroot()

def build_inverted_index():
    global collection_length, N
    for doc in root.findall("doc"):
        doc_id = doc.find("docno").text.strip()
        text = (doc.find("title").text or "") + " " + (doc.find("text").text or "")
        tokens = text.lower().split()
        doc_length[doc_id] = len(tokens)
        dls[doc_id] = len(tokens)
        N += 1
        
        term_counts = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
            collection_term_counts[term] += 1
            collection_length += 1
        
        for term, freq in term_counts.items():
            inverted_index[term][doc_id] = freq

build_inverted_index()

# Load Queries
query_tree = ET.parse(QUERY_PATH)
query_root = query_tree.getroot()
for doc in query_root.findall("top"):
    query_id = doc.find("num").text.strip()
    query_text = doc.find("title").text.strip()
    queriesdict[query_id] = query_text.lower().split()

# BM25 Implementation
def get_kval(dID, k1=1.2, b=0.75):
    avdl = sum(dls.values()) / len(dls)
    dl = dls[dID]
    return k1 * ((1 - b) + (b * (float(dl) / avdl)))

def BM25(dID, ni, fi, qfi, k1=1.2, k2=100, b=0.75):
    ri = 0
    R = 0
    kval = get_kval(dID, k1, b)
    p1 = (((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5)))
    p2 = (((k1 + 1) * fi) / (kval + fi))
    p3 = (((k2 + 1) * qfi) / (k2 + qfi))
    score = math.log(p1+1) * p2 * p3
    return score

def search_bm25():
    results = []
    for query_id, query_terms in queriesdict.items():
        scoredict = defaultdict(float)
        qtermdict = defaultdict(int)
        for term in query_terms:
            qtermdict[term] += 1
        for term, qfi in qtermdict.items():
            if term in inverted_index:
                for docID, fi in inverted_index[term].items():
                    scoredict[docID] += BM25(docID, len(inverted_index[term]), fi, qfi)
        sorted_docs = sorted(scoredict.items(), key=operator.itemgetter(1), reverse=True)
        results.append((query_id, sorted_docs))
    return results

def write_results(results, output_path):
    with open(output_path, "w") as f:
        for query_id, ranked_docs in results:
            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):
                if score > 0:
                    score = 1
                else:
                    score = 0
                f.write(f"{query_id} 0 {doc_id} {rank} {int(score)} BM25\n")

# Run BM25 Search and Save Results
bm25_results = search_bm25()
write_results(bm25_results, OUTPUT_BM25)
print(f"BM25 results saved in {OUTPUT_BM25}")

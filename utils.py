import fitz
import nltk
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [
        w for w in words
        if re.match(r'^[a-zA-Z]+$', w)
        and w not in stopwords.words('indonesian')
    ]
    return words

def build_cooccurrence(words, window_size=2):
    co_occurrences = defaultdict(Counter)

    for i, word in enumerate(words):
        for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
            if i != j:
                co_occurrences[word][words[j]] += 1

    unique_words = list(set(words))
    word_index = {w: i for i, w in enumerate(unique_words)}

    matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            matrix[word_index[word]][word_index[neighbor]] = count

    return matrix, unique_words

def build_graph(matrix):
    return nx.from_numpy_array(matrix)

def centrality_analysis(G):
    return {
        "PageRank": nx.pagerank(G),
        "Degree Centrality": nx.degree_centrality(G),
        "Betweenness Centrality": nx.betweenness_centrality(G)
    }

def draw_graph(G, title, st):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2000)
    plt.title(title)
    st.pyplot(plt)

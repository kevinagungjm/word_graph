import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from nltk.tokenize import sent_tokenize

from utils import (
    extract_text_from_pdf,
    preprocess_text,
    build_cooccurrence,
    build_graph,
    centrality_analysis
)

st.set_page_config(page_title="Word Graph & Centrality", layout="wide")

st.title("📄 Word Graph & Centrality Analysis")

uploaded_file = st.file_uploader("Upload Paper (PDF)", type=["pdf"])

# ======================================================
# SEMUA PROSES HARUS DI DALAM BLOK INI
# ======================================================
if uploaded_file is not None:

    # 1. Ekstraksi teks
    text = extract_text_from_pdf(uploaded_file)

    # 2. Tokenisasi kalimat (untuk statistik)
    sentences = sent_tokenize(text, language='indonesian')

    # 3. Preprocessing kata
    words = preprocess_text(text)

    # Proteksi kalau teks terlalu pendek
    if len(words) < 10:
        st.warning("Dokumen terlalu pendek untuk dianalisis.")
        st.stop()

    # 4. Co-occurrence matrix
    matrix, unique_words = build_cooccurrence(words)

    # 5. GRAPH (INI YANG HILANG DI KODEMU)
    G = build_graph(matrix)

    # =========================
    # STATISTIK
    # =========================
    st.subheader("📊 Statistik Teks & Graph")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Kalimat", len(sentences))
        st.metric("Jumlah Kata Total", len(words))
        st.metric("Jumlah Kata Unik", len(unique_words))

    with col2:
        st.metric("Jumlah Node", G.number_of_nodes())
        st.metric("Jumlah Edge", G.number_of_edges())

    # =========================
    # WORD GRAPH
    # =========================
    st.subheader("📌 Word Graph")

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    nx.draw(G, pos, node_size=150, with_labels=False)
    st.pyplot(plt)

    # =========================
    # CENTRALITY ANALYSIS
    # =========================
    st.subheader("📊 Centrality Analysis")

    centralities = centrality_analysis(G)

    for name, values in centralities.items():
        st.write(f"### {name}")

        df = pd.DataFrame(
            [(node, unique_words[node], score) for node, score in values.items()],
            columns=["Node", "Kata", "Score"]
        )

        df = df.sort_values("Score", ascending=False)
        st.dataframe(df.head(10))

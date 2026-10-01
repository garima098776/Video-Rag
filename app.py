import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

df = joblib.load("embeddings.joblib")
emb_matrix = np.array(df["embedding"].tolist(), dtype=np.float32)

st.title("RAG Video Assistant")

query = st.text_input("Ask your question")

if query:

    q_emb = create_embedding([query])[0]

    sims = cosine_similarity(emb_matrix, [q_emb]).flatten()
    top_idx = sims.argsort()[::-1][:5]

    result_df = df.iloc[top_idx]

    best = result_df.iloc[0]

    start = format_time(best["start"])
    end = format_time(best["end"])

    st.subheader("Best Match")

    st.success(f"{best['number']} | {start} - {end}")
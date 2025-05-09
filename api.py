# app.py - Application Streamlit utilisant les fichiers réduits

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import os

# === Chargement des données ===
@st.cache_data
def load_data():
    clicks_df = pd.read_csv("data/clicks_sample_small.csv")
    articles_df = pd.read_csv("data/articles_metadata_small.csv")
    with open("data/articles_embeddings_small.pickle", "rb") as f:
        embeddings = pickle.load(f)

    # Construction de la matrice utilisateur-article
    user_item_matrix = clicks_df.pivot_table(index='user_id', columns='click_article_id', aggfunc='size', fill_value=0)
    return clicks_df, articles_df, embeddings, user_item_matrix

clicks_df, articles_df, embeddings, user_item_matrix = load_data()

# === Fonctions de recommandation ===
def build_user_profile(user_id, clicks, embedding_dict):
    clicked_ids = clicks[clicks['user_id'] == user_id]['click_article_id']
    vectors = [embedding_dict[aid] for aid in clicked_ids if aid in embedding_dict]
    return np.mean(vectors, axis=0) if vectors else None

def recommend_by_similarity(profile, embedding_dict, top_n=5):
    if profile is None:
        return []
    similarities = {
        aid: cosine_similarity(profile.reshape(1, -1), vec.reshape(1, -1))[0][0]
        for aid, vec in embedding_dict.items()
    }
    top = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [aid for aid, _ in top]

def recommend_by_collab(user_id, matrix, top_n=5):
    if user_id not in matrix.index:
        return []
    seen = matrix.loc[user_id]
    seen = seen[seen > 0].index.tolist()
    if not seen:
        return []
    scores = pd.Series(0, index=matrix.columns)
    for article in seen:
        scores += matrix[article]
    scores = scores.drop(labels=seen)
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def get_similar_articles(article_id, embedding_dict, top_n=5):
    if article_id not in embedding_dict:
        return []
    target = embedding_dict[article_id].reshape(1, -1)
    sims = {
        aid: cosine_similarity(target, vec.reshape(1, -1))[0][0]
        for aid, vec in embedding_dict.items() if aid != article_id
    }
    return [aid for aid, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]]

# === Interface Streamlit ===
st.title(" Système de Recommandation d'Articles")

page = st.sidebar.radio("Navigation", [
    "Exploration", "Recommandation", "Vecteurs", "Comparer"
])

if page == "Exploration":
    st.header(" Exploration des données")
    st.write("### Aperçu des utilisateurs")
    st.dataframe(clicks_df.head())
    st.write("### Aperçu des articles")
    st.dataframe(articles_df.head())

elif page == "Recommandation":
    method = st.radio("Méthode de recommandation", ["Collaboratif", "Embeddings", "Similarité d'article"])
    recos = []

    if method == "Collaboratif":
        user_id = st.selectbox("Choisir un utilisateur", user_item_matrix.index)
        recos = recommend_by_collab(user_id, user_item_matrix)
    elif method == "Embeddings":
        user_id = st.selectbox("Choisir un utilisateur", clicks_df['user_id'].unique())
        profile = build_user_profile(user_id, clicks_df, embeddings)
        recos = recommend_by_similarity(profile, embeddings)
    else:
        article_id = st.selectbox("Choisir un article", articles_df['article_id'].unique())
        recos = get_similar_articles(article_id, embeddings)

    if recos:
        st.success("Articles recommandés :")
        st.dataframe(articles_df[articles_df['article_id'].isin(recos)][['article_id', 'category_id']])
    else:
        st.warning("Aucune recommandation disponible pour ce choix.")

elif page == "Vecteurs":
    st.header(" Vecteur d'un article")
    article_id = st.selectbox("Article à explorer", list(embeddings.keys())[:100])
    vecteur = embeddings.get(article_id)
    if vecteur is not None:
        st.line_chart(vecteur)
    else:
        st.warning("Aucun vecteur disponible.")

elif page == "Comparer":
    st.header(" Comparaison des méthodes")
    user_id = st.selectbox("Choisir un utilisateur", user_item_matrix.index)
    collab = recommend_by_collab(user_id, user_item_matrix)
    profile = build_user_profile(user_id, clicks_df, embeddings)
    embed = recommend_by_similarity(profile, embeddings) if profile is not None else []
    st.subheader("Collaboratif")
    st.dataframe(articles_df[articles_df['article_id'].isin(collab)][['article_id']])
    st.subheader("Embeddings")
    st.dataframe(articles_df[articles_df['article_id'].isin(embed)][['article_id']])
    overlap = set(collab) & set(embed)
    st.markdown(f"###  Overlap : {len(overlap)} article(s) commun(s)")
    if overlap:
        st.dataframe(articles_df[articles_df['article_id'].isin(overlap)][['article_id']])

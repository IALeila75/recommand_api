# app.py - Application Streamlit pour la recommandation

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
    clicks_df = pd.read_csv("data/clicks_sample.csv")
    articles_df = pd.read_csv("data/articles_metadata.csv")
    with open("data/articles_embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)
    return clicks_df, articles_df, embeddings

clicks_df, articles_df, embeddings = load_data()

# === Construction du mapping embeddings ===
if isinstance(embeddings, np.ndarray):
    article_ids = articles_df['article_id'].tolist()
    article_id_to_embedding = {aid: embeddings[i] for i, aid in enumerate(article_ids)}
else:
    article_id_to_embedding = embeddings

# === Création matrice utilisateur × article ===
user_item_matrix = clicks_df.pivot_table(index='user_id', columns='click_article_id', aggfunc='size', fill_value=0)

# === Fonctions de recommandation ===
def build_user_profile(user_id, clicks, embedding_dict):
    clicked_ids = clicks[clicks['user_id'] == user_id]['click_article_id']
    vectors = [embedding_dict[aid] for aid in clicked_ids if aid in embedding_dict]
    return np.mean(vectors, axis=0) if vectors else None

def recommend_by_similarity(profile, embedding_dict, top_n=5):
    if profile is None:
        return []
    scores = {
        aid: cosine_similarity(profile.reshape(1, -1), emb.reshape(1, -1))[0][0]
        for aid, emb in embedding_dict.items()
    }
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [aid for aid, _ in top]

def recommend_by_collab(user_id, matrix, top_n=5):
    if user_id not in matrix.index:
        return []
    seen = matrix.loc[user_id]
    seen = seen[seen > 0].index.tolist()
    scores = pd.Series(0, index=matrix.columns)
    for article in seen:
        scores += matrix[article]
    scores = scores.drop(labels=seen)   
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_popular(clicks, top_n=5):
    return clicks['click_article_id'].value_counts().head(top_n).index.tolist()

def get_similar_articles(article_id, embedding_dict, top_n=5):
    if article_id not in embedding_dict:
        return []
    target_vector = embedding_dict[article_id].reshape(1, -1)
    similarities = {
        aid: cosine_similarity(target_vector, emb.reshape(1, -1))[0][0]
        for aid, emb in embedding_dict.items() if aid != article_id
    }
    top = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top

# === Interface Streamlit ===
st.title("🧠 Système de Recommandation")

page = st.sidebar.radio("Navigation", ["Exploration", "Recommandations"])

if page == "Exploration":
    st.subheader("📊 Exploration des données")
    st.write(clicks_df.head())
    st.write(articles_df.head())
    st.write("Aperçu embeddings :", list(article_id_to_embedding.items())[0])

elif page == "Recommandations":
    method = st.radio("Choisir une méthode", ["Collaboratif", "Embeddings", "Similarité d'article"])
    result_df = pd.DataFrame()

    if method in ["Collaboratif", "Embeddings"]:
        valid_users = [
            uid for uid in clicks_df['user_id'].unique()
            if any(aid in article_id_to_embedding for aid in clicks_df[clicks_df['user_id'] == uid]['click_article_id'])
        ]
        user_id = st.selectbox("Utilisateur :", valid_users)  
        st.write("Articles cliqués :", clicks_df[clicks_df['user_id'] == user_id]['click_article_id'].tolist())

        if method == "Collaboratif":
            recos = recommend_by_collab(user_id, user_item_matrix)
        else:
            profile = build_user_profile(user_id, clicks_df, article_id_to_embedding)
            recos = recommend_by_similarity(profile, article_id_to_embedding)

    else:
        article_id = st.selectbox("Article de référence :", articles_df['article_id'].unique())
        recos = [aid for aid, _ in get_similar_articles(article_id, article_id_to_embedding)]

    if recos:
        st.subheader("📄 Articles recommandés")
        result_df = articles_df[articles_df['article_id'].isin(recos)][['article_id', 'category_id']]
        st.dataframe(result_df)
    else:
        st.warning("Aucune recommandation trouvée pour ce profil.")

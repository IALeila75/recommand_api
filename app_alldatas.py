import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import os

import requests

# === Chargement des données ===
@st.cache_data
def load_data():
    data_path = "data"
    clicks_df = pd.read_csv(os.path.join(data_path, "clicks_sample.csv"))
    articles_df = pd.read_csv(os.path.join(data_path, "articles_metadata.csv"))
    file_path = os.path.join(data_path, "articles_embeddings.pickle")
    
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        print("Téléchargement du fichier...")
        url = "https://www.dropbox.com/scl/fi/zj8udj1gw28j0ebkoi80u/articles_embeddings.pickle?rlkey=dc6ux4r6io1gp205v5hj5378w&st=vcbkrlse&dl=1"  # Remplacez par votre lien direct
        response = requests.get(url)
        if response.content.startswith(b"<html"):
             raise ValueError("❌ Lien Dropbox invalide — page HTML reçue au lieu d’un fichier.")    
        with open(file_path, "wb") as f:
            f.write(response.content)

    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)

    return clicks_df, articles_df, embeddings

clicks_df, articles_df, embeddings = load_data()



# === Construction du mapping embeddings ===
if isinstance(embeddings, np.ndarray):
    article_ids = articles_df['article_id'].tolist()
    article_id_to_embedding = {aid: embeddings[i] for i, aid in enumerate(article_ids)}
else:
    article_id_to_embedding = embeddings

# Matrice utilisateur × article
user_item_matrix = clicks_df.pivot_table(index='user_id', columns='click_article_id', aggfunc='size', fill_value=0)

# === Fonctions utiles ===
def build_user_profile(user_id, clicks, embedding_dict):
    clicked_ids = clicks[clicks['user_id'] == user_id]['click_article_id']
    vectors = [embedding_dict[aid] for aid in clicked_ids if aid in embedding_dict]
    return np.mean(vectors, axis=0) if vectors else None

def recommend_by_similarity(profile, embedding_dict, top_n=5, return_scores=False):
    scores = {
        aid: cosine_similarity(profile.reshape(1, -1), emb.reshape(1, -1))[0][0]
        for aid, emb in embedding_dict.items()
    }
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top if return_scores else [aid for aid, _ in top]

def recommend_by_collab(user_id, matrix, top_n=5):
    if user_id not in matrix.index:
        return []
    user_row = matrix.loc[user_id]
    seen = user_row[user_row > 0].index.tolist()
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

if 'title' not in articles_df.columns:
    articles_df['title'] = 'Article catégorie ' + articles_df['category_id'].astype(str)

def display_articles(article_ids_with_scores):
    if isinstance(article_ids_with_scores[0], tuple):
        article_ids = [aid for aid, _ in article_ids_with_scores]
        df = articles_df[articles_df['article_id'].isin(article_ids)][['article_id', 'title']]
        score_map = dict(article_ids_with_scores)
        df['similarity_score'] = df['article_id'].map(score_map)
        return df.reset_index(drop=True).sort_values(by='similarity_score', ascending=False)
    else:
        return articles_df[articles_df['article_id'].isin(article_ids_with_scores)][['article_id', 'title']].reset_index(drop=True)

# === Interface Streamlit ===
st.title("🧠 Système de Recommandation")

page = st.sidebar.radio("Choisir une vue", ["Exploration & Features", "Recommandations", "Explorer les vecteurs", "Comparer les méthodes"])

if page == "Exploration & Features":
    st.header("📊 Exploration des données & Feature Engineering")

    st.subheader("Aperçu des fichiers")
    with st.expander("📄 clicks_sample.csv"):
        st.write(clicks_df.head())
    with st.expander("📄 articles_metadata.csv"):
        st.write(articles_df.head())
    with st.expander("📦 articles_embeddings.pickle"):
        st.write("Type d'objet:", type(embeddings))
        if isinstance(embeddings, np.ndarray):
            st.write("Dimensions:", embeddings.shape)
            st.write("Premier vecteur:", embeddings[0])
        elif isinstance(embeddings, dict):
            example_key = next(iter(embeddings))
            st.write("Nombre d'articles:", len(embeddings))
            st.write("Exemple d'embedding:", embeddings[example_key])

    st.subheader("✨ Feature Engineering")
    st.markdown("- **click_hour** : heure du clic à partir de `click_timestamp`")
    st.markdown("- **click_day** : jour de la semaine du clic")
    st.markdown("- **session_size** : taille de session utilisateur")

    clicks_df['click_timestamp'] = pd.to_datetime(clicks_df['click_timestamp'], unit='ms')
    clicks_df['click_hour'] = clicks_df['click_timestamp'].dt.hour
    clicks_df['click_day'] = clicks_df['click_timestamp'].dt.dayofweek

    st.write(clicks_df[['user_id', 'click_article_id', 'click_timestamp', 'click_hour', 'click_day']].head())

    st.subheader("📈 Diagrammes de distribution")
    clicks_per_article = clicks_df['click_article_id'].value_counts()
    fig1, ax1 = plt.subplots()
    sns.histplot(clicks_per_article, bins=50, ax=ax1)
    ax1.set_title('Distribution des clics par article')
    st.pyplot(fig1)

    clicks_per_user = clicks_df['user_id'].value_counts()
    fig2, ax2 = plt.subplots()
    sns.histplot(clicks_per_user, bins=50, ax=ax2)
    ax2.set_title('Distribution des clics par utilisateur')
    st.pyplot(fig2)

    st.subheader("☁️ Nuage de mots des titres")
    titles_text = " ".join(articles_df['title'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

elif page == "Recommandations":
    st.header("🔎 Recommandations personnalisées")

    method = st.radio("Méthode de recommandation :", ["Collaboratif", "Embeddings", "Similarité d'article"])

    recos = []
    result_df = pd.DataFrame()

    if method == "Collaboratif" or method == "Embeddings":
        user_id = st.selectbox("Sélectionnez un utilisateur :", clicks_df['user_id'].unique())

        if method == "Collaboratif":
            recos = recommend_by_collab(user_id, user_item_matrix)
        else:
            profile = build_user_profile(user_id, clicks_df, article_id_to_embedding)
            recos = recommend_by_similarity(profile, article_id_to_embedding, return_scores=False) if profile is not None else []

        result_df = display_articles(recos)

    elif method == "Similarité d'article":
        article_id = st.selectbox("Choisissez un article :", articles_df['article_id'].unique())
        recos = get_similar_articles(article_id, article_id_to_embedding, top_n=5)
        result_df = display_articles(recos)

    if result_df.empty:
        st.warning("Aucune recommandation disponible. Voici des articles populaires :")
        recos = recommend_popular(clicks_df)
        result_df = display_articles(recos)

    st.subheader("📋 Résultats des recommandations")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les recommandations",
        data=csv,
        file_name="recommandations.csv",
        mime='text/csv'
    )

elif page == "Explorer les vecteurs":
    st.header("🔬 Visualisation d’un vecteur d’article")

    article_id = st.selectbox("Choisir un article :", list(article_id_to_embedding.keys())[:500])
    
    if article_id in article_id_to_embedding:
        vecteur = article_id_to_embedding[article_id]
        st.write(f"Vecteur de l’article {article_id} :")
        st.write(vecteur)
        st.line_chart(vecteur)
    else:
        st.warning("Aucun vecteur trouvé pour cet article.")

elif page == "Comparer les méthodes":
    st.header("📊 Comparaison des Méthodes")

    user_id = st.selectbox("Sélectionnez un utilisateur :", clicks_df['user_id'].unique())

    collab = recommend_by_collab(user_id, user_item_matrix)
    profile = build_user_profile(user_id, clicks_df, article_id_to_embedding)
    embed = recommend_by_similarity(profile, article_id_to_embedding) if profile is not None else []

    st.subheader("📘 Recommandations - Filtrage Collaboratif")
    st.dataframe(display_articles(collab))

    st.subheader("📙 Recommandations - Embeddings")
    st.dataframe(display_articles(embed))

    overlap = set(collab).intersection(set(embed))
    st.markdown(f"### ✅ Overlap : {len(overlap)} article(s) en commun")
    if overlap:
        st.dataframe(display_articles(list(overlap)))

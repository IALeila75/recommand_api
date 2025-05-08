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

import requests

# === Chargement des données ===

@st.cache_data
def load_data():
    data_path = "data"
    os.makedirs(data_path, exist_ok=True)

    # Chargement initial
    clicks_df = pd.read_csv(os.path.join(data_path, "clicks_sample.csv"))
    articles_df = pd.read_csv(os.path.join(data_path, "articles_metadata.csv"))

    # Filtrage intelligent : conserver les utilisateurs les plus actifs
    top_users = clicks_df['user_id'].value_counts().head(50).index
    clicks_df = clicks_df[clicks_df['user_id'].isin(top_users)]

    file_path = os.path.join(data_path, "articles_embeddings_small.pickle")

    # Téléchargement si le fichier n’existe pas
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        url = "https://www.dropbox.com/scl/fi/722ihff4354i8qelo4pqn/articles_embeddings_small.pickle?rlkey=9k1eqoqt3lzumrp9cl88sczuj&st=2qcfbg3e&dl=1"
        response = requests.get(url)
        if response.content.startswith(b"<html"):
            raise ValueError("❌ Le lien Dropbox ne fournit pas un fichier binaire valide.")
        with open(file_path, "wb") as f:
            f.write(response.content)

    # Chargement sécurisé du pickle
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)

    # Réduction de la taille du dictionnaire d'embeddings
    if isinstance(embeddings, np.ndarray):
        article_ids = articles_df['article_id'].tolist()
        embeddings = {aid: embeddings[i] for i, aid in enumerate(article_ids)}
    else:
        embeddings = {aid: vec for aid, vec in embeddings.items() if aid in articles_df['article_id'].values}

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
    
if 'title' not in articles_df.columns:
    articles_df['title'] = 'Article catégorie ' + articles_df['category_id'].astype(str)

def display_articles(article_ids_with_scores):
    if not article_ids_with_scores:
        return pd.DataFrame(columns=["article_id", "title", "similarity_score"])

    if isinstance(article_ids_with_scores[0], tuple):
        article_ids = [aid for aid, _ in article_ids_with_scores]
        df = articles_df[articles_df['article_id'].isin(article_ids)][['article_id', 'title']]
        score_map = dict(article_ids_with_scores)
        df['similarity_score'] = df['article_id'].map(score_map)
        return df.reset_index(drop=True).sort_values(by='similarity_score', ascending=False)
    else:
        return articles_df[articles_df['article_id'].isin(article_ids_with_scores)][['article_id', 'title']].reset_index(drop=True)

def evaluate_collaborative(clicks_df, matrix, k=5):
    hits, total = 0, 0
    for user_id in clicks_df['user_id'].unique():
        true_articles = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'])
        recos = recommend_by_collab(user_id, matrix, top_n=k)
        st.write("🪪 Recommandations collaboratives :", collab)


        if any(a in true_articles for a in recos):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

def evaluate_similarity(clicks_df, embedding_dict, k=5):
    hits, total = 0, 0
    for user_id in clicks_df['user_id'].unique():
        profile = build_user_profile(user_id, clicks_df, embedding_dict)
        if profile is None:
            continue
        true_articles = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'])
        recos = recommend_by_similarity(profile, embedding_dict, top_n=k)
        if any(a in true_articles for a in recos):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0

def evaluate_popular(clicks_df, k=5):
    popular_articles = recommend_popular(clicks_df, top_n=k)
    hits, total = 0, 0
    for user_id in clicks_df['user_id'].unique():
        true_articles = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'])
        if any(a in true_articles for a in popular_articles):
            hits += 1
        total += 1
    return hits / total if total > 0 else 0
    
    
# === Fonction recommandation collaborative corrigée ===
def recommend_by_collab(user_id, matrix, top_n=5):
    if user_id not in matrix.index:
        return []

    # Articles déjà vus par l'utilisateur
    seen = matrix.loc[user_id]
    seen = seen[seen > 0].index.tolist()

    # Somme des scores de tous les utilisateurs pour chaque article
    scores = matrix.sum(axis=0)

    # On retire les articles déjà vus
    scores = scores.drop(seen, errors='ignore')

    # Retourne les top-N articles non vus les plus populaires
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

# === Construction correcte de la matrice utilisateur × article ===
user_item_matrix = clicks_df.pivot_table(
    index='user_id',
    columns='click_article_id',
    aggfunc='size',
    fill_value=0
)
 
# === Interface Streamlit ===
st.title("🧠 Système de Recommandation")

page = st.sidebar.radio("Choisir une vue", [
    "Exploration & Features", 
    "Explorer les vecteurs",
    "Recommandations",      
    "Comparer les méthodes", 
    "Évaluation"  # 👈 nouveau menu
])

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
        
elif page == "Recommandations":
    method = st.radio("Choisir une méthode", ["Collaboratif", "Embeddings", "Similarité d'article"])
    result_df = pd.DataFrame()

    if method == "Collaboratif":
        valid_users = user_item_matrix.index.tolist()
        if not valid_users:
            st.warning("❌ Aucun utilisateur disponible pour la recommandation collaborative.")
        else:
            user_id = st.selectbox("Utilisateur (collaboratif) :", valid_users)
            recos = recommend_by_collab(user_id, user_item_matrix)
            st.write("🔁 Recommandations collaboratives :", recos)

    elif method == "Embeddings":
        valid_users_embed = [
            uid for uid in clicks_df['user_id'].unique()
            if build_user_profile(uid, clicks_df, article_id_to_embedding) is not None
        ]
        if not valid_users_embed:
            st.warning("❌ Aucun utilisateur valide pour la recommandation par embeddings.")
            recos = []
        else:
            user_id = st.selectbox("Utilisateur (embeddings) :", valid_users_embed)
            profile = build_user_profile(user_id, clicks_df, article_id_to_embedding)
            if profile is None:
                st.warning("⚠️ Profil utilisateur introuvable.")
                recos = []
            else:
                recos = recommend_by_similarity(profile, article_id_to_embedding)
                st.write("🔁 Recommandations par embeddings :", recos)

    else:  # Similarité d'article
        article_id = st.selectbox("Article de référence :", articles_df['article_id'].unique())
        recos = [aid for aid, _ in get_similar_articles(article_id, article_id_to_embedding)]
        st.write("🔁 Articles similaires :", recos)

    if recos:
        st.subheader("📄 Articles recommandés")
        result_df = articles_df[articles_df['article_id'].isin(recos)][['article_id', 'title', 'category_id']]
        st.dataframe(result_df)
    else:
        st.warning("❌ Aucune recommandation trouvée pour ce profil.")



elif page == "Comparer les méthodes":
    st.header("📊 Comparaison des Méthodes")

    user_id = st.selectbox("Sélectionnez un utilisateur :", clicks_df['user_id'].unique())

    collab = recommend_by_collab(user_id, user_item_matrix)
    st.write("🪪 Recommandations brutes :", recos)

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

elif page == "Évaluation":
    st.header("📊 Évaluation des Méthodes de Recommandation")

    k = st.slider("🎯 Choisissez la valeur de k (top-k)", 1, 20, 5)

    if st.button("Lancer l'évaluation"):
        with st.spinner("Évaluation en cours..."):
            recall_collab = evaluate_collaborative(clicks_df, user_item_matrix, k=k)
            recall_sim = evaluate_similarity(clicks_df, article_id_to_embedding, k=k)
            recall_pop = evaluate_popular(clicks_df, k=k)

        # Résumé sous forme de DataFrame
        results_df = pd.DataFrame({
            "Méthode": ["Collaboratif", "Embeddings", "Populaire"],
            "Recall": [recall_collab, recall_sim, recall_pop]
        })
        #sns.barplot(data=results_df, x="Méthode", y="Recall", palette="pastel", ax=ax)

        # Affichage tabulaire
        st.markdown("### 📋 Résultats des évaluations")
        st.dataframe(results_df)

        # Sauvegarde CSV
        csv_path = f"evaluation_results_k{k}.csv"
        results_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger les résultats CSV",
                data=f,
                file_name=csv_path,
                mime='text/csv'
            )

        # Barplot
        st.markdown("### 📊 Comparaison graphique")
        fig, ax = plt.subplots()
        sns.barplot( data=results_df,    x="Méthode",
        y="Recall", hue="Méthode", palette="pastel", ax=ax,
        legend=False)
        ax.set_ylim(0, 1)
        ax.set_title(f"Recall@{k} par méthode")
        st.pyplot(fig)


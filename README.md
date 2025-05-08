#  Système de Recommandation d'Articles

Ce projet est une application Streamlit de recommandation de contenu basée sur :

-  Filtrage collaboratif
-  Similarité d'articles via embeddings
-  Représentation d'utilisateurs dans un espace latent

---

##  Structure du projet

```
.
├── app.py                         # Application principale Streamlit
├── requirements.txt              # Dépendances
├── .gitignore                    # Fichiers exclus du dépôt
├── data/                         # Fichiers de données (non inclus dans le repo)
│   ├── clicks_sample.csv
│   ├── articles_metadata.csv
│   └── articles_embeddings.pickle
```

---

##  Lancer l'application

1. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate    # Windows
```

2. Installer les dépendances

```bash
pip install -r requirements.txt
```

3. Lancer l'app

```bash
streamlit run app.py
```

---

##  Données

 Les fichiers suivants ne sont **pas stockés dans ce dépôt** car ils dépassent les 100 Mo :

- `data/articles_embeddings.pickle`
- `data/clicks_sample.csv`
- `data/articles_metadata.csv`

Vous devez les placer manuellement dans le dossier `data/` pour que l’application fonctionne.

---

##  Fonctionnalités de l'app

- Vue exploration : données, distributions, nuage de mots
- Recommandation :
  - par filtrage collaboratif
  - par similarité d'embeddings
  - par similarité d’article
- Visualisation de vecteurs d’articles
- Comparaison des méthodes

---

##  Auteur

Projet réalisé par [Votre nom] dans le cadre du projet IA – Recommandation.
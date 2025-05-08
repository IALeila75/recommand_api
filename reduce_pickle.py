import pickle
import pandas as pd

# Charger les articles utilisés dans l'échantillon
articles_df = pd.read_csv("data/articles_metadata.csv").head(1000)
article_ids = set(articles_df["article_id"])

# Charger le gros fichier pickle
with open("data/articles_embeddings.pickle", "rb") as f:
    embeddings = pickle.load(f)

# Réduire le dico des embeddings uniquement à ceux nécessaires
if isinstance(embeddings, dict):
    reduced_embeddings = {aid: emb for aid, emb in embeddings.items() if aid in article_ids}
else:
    # embeddings = np.ndarray
    reduced_embeddings = {aid: embeddings[i] for i, aid in enumerate(articles_df["article_id"])}

# Sauvegarder un nouveau fichier .pickle
with open("data/articles_embeddings_small.pickle", "wb") as f:
    pickle.dump(reduced_embeddings, f)

print("✅ Fichier réduit sauvegardé : articles_embeddings_small.pickle")

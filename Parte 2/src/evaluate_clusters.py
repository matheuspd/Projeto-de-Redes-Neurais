import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

OUTPUT_FOLDER = "../clip_outputs"

print("[INFO] Carregando dados...")

# Carrega embeddings e labels salvos
data = np.load(f"{OUTPUT_FOLDER}/clip_embeddings_pca.npz")
df_filenames = pd.read_csv(f"{OUTPUT_FOLDER}/clip_train_filenames.csv")

embeddings_kmeans = data["embeddings"]
embeddings_hdbscan = data["embeddings_hdbscan"]
kmeans_labels = data["kmeans"]
hdbscan_labels = data["hdbscan"]

# Adiciona os labels ao DataFrame
df_filenames["kmeans_150_label"] = kmeans_labels
df_filenames["hdbscan_label"] = hdbscan_labels

# Dicionário com modelos e dados
modelos = {
    "clip_kmeans_150": ("kmeans_150_label", embeddings_kmeans),
    "clip_hdbscan": ("hdbscan_label", embeddings_hdbscan),
}

print("\n=== Avaliação dos clusters CLIP ===\n")
for nome_modelo, (col_label, X_full) in modelos.items():
    print(f"Avaliando: {nome_modelo}")

    labels = df_filenames[col_label].values
    mask_validos = labels != -1  # Exclui ruído do HDBSCAN
    X = X_full[mask_validos]
    y = labels[mask_validos]

    n_clusters = len(set(y))
    if -1 in y:
        n_clusters -= 1

    if n_clusters < 2:
        print("  [AVISO] Clusters insuficientes para avaliação.")
        continue

    print(f"  Clusters válidos: {n_clusters}")

    # Silhouette Score
    try:
        silh = silhouette_score(X, y)
        print(f"  Silhouette Score: {silh:.4f}")
    except Exception as e:
        print(f"  [ERRO] Silhouette Score: {e}")

    # Davies-Bouldin Index
    try:
        db = davies_bouldin_score(X, y)
        print(f"  Davies-Bouldin Index: {db:.4f}")
    except Exception as e:
        print(f"  [ERRO] Davies-Bouldin Index: {e}")

    # Calinski-Harabasz Index
    try:
        ch = calinski_harabasz_score(X, y)
        print(f"  Calinski-Harabasz Index: {ch:.4f}")
    except Exception as e:
        print(f"  [ERRO] Calinski-Harabasz Index: {e}")

    print()

print("[INFO] Avaliação completa.")

"""
Explicar resultados:

Avaliando: clip_kmeans_150     
  Clusters válidos: 150
  Silhouette Score: 0.0414
  Davies-Bouldin Index: 3.5961
  Calinski-Harabasz Index: 85.3613

Avaliando: clip_hdbscan       
  Clusters válidos: 49
  Silhouette Score: 0.1431
  Davies-Bouldin Index: 1.5689
  Calinski-Harabasz Index: 109.9561
"""

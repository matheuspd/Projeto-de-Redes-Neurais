import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = "../models"
CSV_PATH = f"{MODELS_DIR}/descricao_clusters.csv"
CSV_HDBSCAN_PATH = f"{MODELS_DIR}/descricao_hdbscan.csv"
HDBSCAN_EMB_PATH = f"{MODELS_DIR}/embeddings_hdbscan.npz"

# Lista de modelos disponíveis
modelos = {
    "kmeans_100": "kmeans_100_label",
    "kmeans_150": "kmeans_150_label",
    "kmeans_200": "kmeans_200_label",
    "hdbscan": "hdbscan_label"
}

print("Carregando dados e modelos...")
df = pd.read_csv(CSV_PATH)
pca_data = np.load(f"{MODELS_DIR}/embeddings_pca.npz")
X_pca = pca_data["embeddings"]
df["embedding_pca"] = list(X_pca)

# Dados amostrados do HDBSCAN
desc_sample = pd.read_csv(CSV_HDBSCAN_PATH)
hdbscan_data = np.load(HDBSCAN_EMB_PATH)
embeddings_hdb = hdbscan_data["embeddings"]
hdbscan_labels = hdbscan_data["labels"]

desc_sample["hdbscan_label"] = hdbscan_labels

# Preenche os rótulos dos modelos no df
for modelo, col_name in modelos.items():
    if modelo == "hdbscan": continue
    if col_name not in df.columns and modelo in pca_data:
        df[col_name] = pca_data[modelo]

print("\nModelos disponíveis:")
for m in modelos:
    print(f"- {m}")

# Escolha do modelo
modelo_escolhido = input("Escolha o modelo (ex: kmeans_100, hdbscan): ").strip()

if modelo_escolhido not in modelos:
    print("Modelo inválido. Encerrando.")
    exit()

col_cluster = modelos[modelo_escolhido]

print(f"Buscando representantes por cluster usando modelo: {modelo_escolhido}...")

descricao_cluster = {}

if modelo_escolhido == "hdbscan":
    # Usar dados amostrados e embeddings do HDBSCAN
    cluster_ids = sorted(c for c in desc_sample[col_cluster].unique() if c != -1)
    for cluster_id in cluster_ids:
        cluster_df = desc_sample[desc_sample[col_cluster] == cluster_id]
        cluster_embs = np.stack(embeddings_hdb[desc_sample[col_cluster] == cluster_id])
        centroide = cluster_embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroide, cluster_embs)[0]
        idx = np.argmax(sims)
        descricao = cluster_df.iloc[idx]['description']
        descricao_cluster[cluster_id] = descricao
else:
    # Para KMeans usar df completo e embeddings PCA
    cluster_ids = sorted(c for c in df[col_cluster].unique() if c != -1)
    for cluster_id in cluster_ids:
        cluster_df = df[df[col_cluster] == cluster_id]
        cluster_embs = np.stack(cluster_df['embedding_pca'].values)
        centroide = cluster_embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroide, cluster_embs)[0]
        idx = np.argmax(sims)
        descricao = cluster_df.iloc[idx]['description']
        descricao_cluster[cluster_id] = descricao

total_clusters = len(descricao_cluster)
print("Total de clusters:", total_clusters)

# Solicita quantos clusters mostrar
try:
    num_clusters = int(input("Mostrar descrição de quantos clusters? "))
    if num_clusters > total_clusters or num_clusters < 1:
        print("Entrada maior do que o número total ou menos que 1. Mostrando 10 clusters.")
        num_clusters = 10
except ValueError:
    print("Entrada inválida. Mostrando 10 clusters.")
    num_clusters = 10

print("\nDescrições representativas por cluster:")
for cid, desc in list(descricao_cluster.items())[:num_clusters]:
    print(f"Cluster {cid}: {desc}")
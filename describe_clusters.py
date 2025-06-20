import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Carregando dados e modelos...")
df = pd.read_csv("descricao_clusters.csv")
pca = joblib.load("model_pca.joblib")
model = SentenceTransformer("model_sentence_transformer")

print("Recomputando embeddings normalizados + PCA...")
embeddings = model.encode(df['description'].tolist(), show_progress_bar=True, convert_to_numpy=True)
embeddings = normalize(embeddings, norm='l2')
X_pca = pca.transform(embeddings)

print("Buscando representantes por cluster...")
df['embedding_pca'] = list(X_pca)

descricao_cluster = {}
for cluster_id in sorted(df['cluster'].unique()):
    cluster_df = df[df['cluster'] == cluster_id]
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
    if num_clusters > total_clusters:
        print("Entrada maior do que o número de clusters total. Mostrando 10 clusters por padrão.")
        num_clusters = 10
except ValueError:
    print("Entrada inválida. Mostrando 10 clusters por padrão.")
    num_clusters = 10

print("Descrições representativas por cluster:")
for cid, desc in list(descricao_cluster.items())[:num_clusters]:
    print(f"Cluster {cid}: {desc}")
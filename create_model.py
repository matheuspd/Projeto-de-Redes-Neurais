import joblib
import warnings
import torch
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("Carregando Dataset...")

# Caminho do CSV (ajuste se necessário)
CSV_PATH = "flickr_annotations_30k.csv"

#  1. Carrega e extrai as descrições 
df = pd.read_csv(CSV_PATH)

rows = []
for _, row in df.iterrows():
    descriptions = literal_eval(row['raw'])  # converte string para lista
    filename = row['filename']
    for desc in descriptions:
        rows.append({'filename': filename, 'description': desc})

desc_df = pd.DataFrame(rows)

# desc_df = desc_df.sample(n=50000, random_state=1337)

print("Dataframe pronto.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

#  2. Gera os embeddings 
print("Gerando embeddings com SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = model.encode(
    desc_df['description'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Normalizando embeddings...")
embeddings = normalize(embeddings, norm='l2')

print("Reduzindo dimensionalidade...")

pca = PCA(n_components=embeddings.shape[1]//2, random_state=1337)
embeddings = pca.fit_transform(embeddings)
print(f"Variância explicada total pelo PCA: {np.sum(pca.explained_variance_ratio_):.2%}")

#  3. Aplica KMeans 
print("Aplicando KMeans...")
num_clusters = 125      # Mudar conforme o necessário
kmeans = KMeans(n_clusters=num_clusters, random_state=1337)
labels = kmeans.fit_predict(embeddings)
desc_df['cluster'] = labels

#  4. Reduz para 2D com UMAP 
print("Aplicando UMAP para visualização 2D...")
umap_visual = umap.UMAP(
    n_neighbors=20,
    min_dist=0.2,
    metric='cosine'
)
embedding_2d = umap_visual.fit_transform(embeddings)
desc_df['umap_x'] = embedding_2d[:, 0]
desc_df['umap_y'] = embedding_2d[:, 1]

#  5. Salva os resultados 
print("Salvando arquivos...")

desc_df.to_csv("descricao_clusters.csv", index=False)
joblib.dump(kmeans, "model_kmeans.joblib")
joblib.dump(pca, "model_pca.joblib")
model.save("model_sentence_transformer")
# Salvar embeddings + labels para avaliação
np.savez("embeddings_labels.npz", embeddings=embeddings, labels=labels)

print("Modelos e resultados salvos.")

#  6. Visualização 
print("Plotando visualização...")
plt.figure(figsize=(12, 8))

# Padrão com 256 cores distintas
palette = cc.glasbey[:num_clusters]
sns.scatterplot(
    x="umap_x", y="umap_y",
    hue="cluster",
    palette=palette,
    data=desc_df,
    legend=None,
    alpha=0.6
)
plt.title("Clusters de descrições via KMeans + UMAP")
plt.savefig("clusters_visualizacao.png", dpi=300, bbox_inches='tight')
plt.show()

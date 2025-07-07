"""
Script para extração de descrições textuais do dataset Flickr30k, geração de embeddings com SentenceTransformer,
redução de dimensionalidade com PCA, clustering com KMeans e HDBSCAN, e visualização dos clusters
com UMAP e t-SNE. Os resultados são salvos para uso posterior.

Dataset: Flickr30k
------------------
O Flickr30k é um conjunto de dados com aproximadamente 30.000 imagens de cenas cotidianas, cada uma anotada com 5
descrições textuais (legendas) escritas por humanos. É amplamente utilizado em tarefas de visão computacional
e PLN, como busca cruzada imagem-texto e geração de legendas automáticas.

Resultado esperado:
-------------------
- Arquivos salvos:
  - CSV com descrições e filenames (`descricao_clusters.csv`)
  - Arquivo `.npz` com embeddings PCA e rótulos de clusters
  - Modelos KMeans, HDBSCAN, PCA e SentenceTransformer serializados
  - Imagens de visualização dos clusters usando UMAP e t-SNE
"""

import os
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
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== Criar pastas para salvar arquivos =====================
os.makedirs("../models", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

# ===================== Carregamento do Dataset =====================
print("Carregando Dataset...")

# Caminho do CSV (ajuste se necessário)
CSV_PATH = "../../flickr_annotations_30k.csv"

# Lê o arquivo CSV contendo as colunas: 'filename' e 'raw' (lista com 5 descrições)
df = pd.read_csv(CSV_PATH)

# Transforma cada elemento do vetor em linhas, cada uma com uma descrição
rows = []
for _, row in df.iterrows():
    descriptions = literal_eval(row['raw'])  # converte string para lista
    filename = row['filename']
    for desc in descriptions:
        rows.append({'filename': filename, 'description': desc})

# Cria novo DataFrame com uma linha por descrição
desc_df = pd.DataFrame(rows)
# desc_df = desc_df.sample(n=10000, random_state=1337)
print("Dataframe pronto.")

# ===================== Geração dos Embeddings =====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Carrega modelo pré-treinado para gerar embeddings das descrições
print("Gerando embeddings com SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = model.encode(
    desc_df['description'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normaliza os vetores para norma L2
print("Normalizando embeddings...")
embeddings = normalize(embeddings, norm='l2')

# ===================== Redução de Dimensionalidade com PCA =====================

# Normalização para PCA
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

joblib.dump(scaler, '../models/model_scaler.joblib')

print("Reduzindo dimensionalidade...")
pca = PCA(n_components=embeddings.shape[1]//3, random_state=1337)
embeddings_reduced = pca.fit_transform(embeddings)

# ===================== Função auxiliar para visualização =====================
def plot_and_save(embeddings_2d, labels, method_name, model_name, desc_subset):
    df_vis = desc_subset.copy()
    df_vis['x'] = embeddings_2d[:, 0]
    df_vis['y'] = embeddings_2d[:, 1]
    df_vis['cluster'] = labels

    num_clusters = len(set(labels))
    palette = cc.glasbey[:num_clusters]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="x", y="y",
        hue="cluster",
        palette=palette,
        data=df_vis,
        legend=None,
        alpha=0.6
    )
    plt.title(f"{model_name} + {method_name}")
    filename = f"../plots/{model_name}_{method_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualização salva: {filename}")

# ===================== Clustering com KMeans =====================
for k in [100, 150, 200]:
    print(f"Treinando KMeans com {k} clusters...")
    kmeans = KMeans(n_clusters=k, random_state=1337)
    labels = kmeans.fit_predict(embeddings_reduced)

    if k == 100:
        kmeans_100_labels = labels
    elif k == 150:
        kmeans_150_labels = labels
    elif k == 200:
        kmeans_200_labels = labels

    # Salva o modelo KMeans treinado
    model_id = f"kmeans_{k}"
    joblib.dump(kmeans, f"../models/{model_id}.joblib")

    # Resample para visualização
    sample_indices = resample(
        np.arange(len(embeddings_reduced)),
        replace=False,
        n_samples=len(embeddings_reduced) // 3,
        stratify=labels,
        random_state=1337
    )

    # Usa os índices para amostrar todos os dados
    desc_sample = None
    embeddings_sample = embeddings_reduced[sample_indices]
    labels_sample = labels[sample_indices]
    desc_sample = desc_df.iloc[sample_indices].reset_index(drop=True)

    # Gerar as visualizações demora um pouco (especialmente t-SNE)
    # UMAP (sem random_state para permitir paralelização)
    print(f"Gerando UMAP para {model_id}...")
    umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    emb_umap = umap_2d.fit_transform(embeddings_sample)
    plot_and_save(emb_umap, labels_sample, "UMAP", model_id, desc_sample)

    # t-SNE
    print(f"Gerando t-SNE para {model_id}...")
    tsne_2d = TSNE(n_components=2, perplexity=6, n_jobs=-1, random_state=1337)
    emb_tsne = tsne_2d.fit(embeddings_sample)
    plot_and_save(emb_tsne, labels_sample, "tSNE", model_id, desc_sample)

# ===================== Clustering com HDBSCAN =====================

print("Reduzindo dimensionalidade dos embeddings originais para HDBSCAN (maior redução)...")
pca_hdbscan = PCA(n_components=30, random_state=1337)
embeddings_hdbscan = pca_hdbscan.fit_transform(embeddings)

# Resample para reduzir tempo de treino (usa apenas 1/3 dos dados)
print("Realizando resample para HDBSCAN...")
sample_indices = resample(
    np.arange(len(embeddings_hdbscan)),
    replace=False,
    n_samples=len(embeddings_hdbscan) // 3,
    random_state=1337
)

# Usa os índices para selecionar subconjunto de dados
desc_sample = None
embeddings_hdb = embeddings_hdbscan[sample_indices]
desc_sample = desc_df.iloc[sample_indices].reset_index(drop=True)

# Treinamento HDBSCAN (demora bastante com muitos dados)
print("Treinando HDBSCAN...")
hdb = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=3,
    prediction_data=True
)
hdbscan_labels = hdb.fit_predict(embeddings_hdb)

# Salva o modelo HDBSCAN
joblib.dump(hdb, "../models/hdbscan.joblib")

# Gerar as visualizações demora um pouco (especialmente t-SNE)
# UMAP (sem random_state para permitir paralelização)
print("Gerando UMAP para HDBSCAN...")
umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
emb_umap = umap_2d.fit_transform(embeddings_hdb)
plot_and_save(emb_umap, hdbscan_labels, "UMAP", "HDBSCAN", desc_sample)

# t-SNE
print("Gerando t-SNE para HDBSCAN...")
tsne_2d = TSNE(n_components=2, perplexity=6, n_jobs=-1, random_state=1337)
emb_tsne = tsne_2d.fit(embeddings_hdb)
plot_and_save(emb_tsne, hdbscan_labels, "tSNE", "HDBSCAN", desc_sample)


# ===================== Salvando os Resultados =====================
print("Salvando arquivos...")
# Salva o dataframe com descrições
desc_df.to_csv("../models/descricao_clusters.csv", index=False)
# Salva os embeddings reduzidos e rótulos de clusters em um .npz
np.savez("../models/embeddings_pca.npz", 
        embeddings=embeddings_reduced,
        kmeans_100=kmeans_100_labels,
        kmeans_150=kmeans_150_labels,
        kmeans_200=kmeans_200_labels,
        )
# Salvar embeddings PCA usados no treinamento do HDBSCAN (amostrados)
np.savez("../models/embeddings_hdbscan.npz", 
        embeddings=embeddings_hdb,
        labels=hdbscan_labels,
        )
# Salva o modelo PCA e o modelo de SentenceTransformer
joblib.dump(pca, "../models/model_pca.joblib")
joblib.dump(pca_hdbscan, "../models/model_pca_hdbscan.joblib")
model.save("../models/model_sentence_transformer")
desc_sample.to_csv("../models/descricao_hdbscan.csv", index=False)
print("Modelos e resultados salvos.")
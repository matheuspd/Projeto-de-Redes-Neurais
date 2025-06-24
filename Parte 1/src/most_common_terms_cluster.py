import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

MODELS_DIR = "../models"

nltk.download("stopwords")
stop_words = list(stopwords.words("english"))

# Carrega embeddings PCA padrão e CSV completo
data = np.load(f"{MODELS_DIR}/embeddings_pca.npz")
embeddings = data['embeddings']
df = pd.read_csv(f"{MODELS_DIR}/descricao_clusters.csv")

# Carrega CSV amostrado do HDBSCAN
desc_sample = pd.read_csv(f"{MODELS_DIR}/descricao_hdbscan.csv")
hdbscan_data = np.load(f"{MODELS_DIR}/embeddings_hdbscan.npz")
embeddings_hdb = hdbscan_data["embeddings"]
hdbscan_labels = hdbscan_data["labels"]

desc_sample["hdbscan_label"] = hdbscan_labels

# Modelos disponíveis e colunas correspondentes
modelos = {
    "kmeans_100": "kmeans_100_label",
    "kmeans_150": "kmeans_150_label",
    "kmeans_200": "kmeans_200_label",
    "hdbscan": "hdbscan_label"
}

# Adiciona colunas de labels no df completo para KMeans
for modelo, col_label in modelos.items():
    if modelo != "hdbscan":  # KMeans: adiciona do .npz
        nome_npz = modelo
        if col_label not in df.columns:
            if nome_npz in data:
                df[col_label] = data[nome_npz]
            else:
                print(f"[AVISO] Chave '{nome_npz}' não encontrada no arquivo .npz; coluna '{col_label}' não adicionada.")


# Para HDBSCAN, garante que desc_sample tem a coluna de label
if "hdbscan_label" not in desc_sample.columns:
    print("[AVISO] Coluna 'hdbscan_label' não encontrada no arquivo de amostra HDBSCAN.")

# Entrada do usuário para limitar clusters
try:
    limite_clusters = int(input("Quantos clusters mostrar por modelo? "))
    if limite_clusters < 1:
        print("Número inválido. Mostrando 10 por padrão.")
        limite_clusters = 10
except ValueError:
    print("Entrada inválida. Mostrando 10 por padrão.")
    limite_clusters = 10

top_k = 10

for nome_modelo, col_label in modelos.items():
    print(f"\n=== Top palavras por cluster ({nome_modelo}) ===")
    
    if nome_modelo == "hdbscan":
        # Usar df amostrado do HDBSCAN
        if col_label not in desc_sample.columns:
            print(f"[ERRO] Coluna '{col_label}' não encontrada no CSV amostrado HDBSCAN. Pulando.")
            continue
        
        df_model = desc_sample[desc_sample[col_label] != -1]  # Remove ruído
    else:
        # Usar df completo para KMeans
        if col_label not in df.columns:
            print(f"[ERRO] Coluna '{col_label}' não encontrada no CSV principal. Pulando.")
            continue
        
        df_model = df[df[col_label] != -1]  # Remove ruído
    
    # Agrupa descrições por cluster
    cluster_groups = df_model.groupby(col_label)["description"].apply(list)
    total_clusters = len(cluster_groups)
    
    if limite_clusters > total_clusters:
        print(f"Modelo {nome_modelo} tem apenas {total_clusters} clusters. Mostrando todos.")
        clusters_a_mostrar = cluster_groups.items()
    else:
        clusters_a_mostrar = list(cluster_groups.items())[:limite_clusters]

    # Junta todas as descrições do cluster em um único documento
    cluster_docs = {
        cluster: " ".join(descriptions)
        for cluster, descriptions in clusters_a_mostrar
    }

    # Cria matriz TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(cluster_docs.values())
    feature_names = vectorizer.get_feature_names_out()

    # Mostra top termos por cluster
    for i, (cluster_id, _) in enumerate(cluster_docs.items()):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        top_indices = tfidf_scores.argsort()[::-1][:top_k]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"Cluster {cluster_id}: {', '.join(top_words)}")
import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from hdbscan import approximate_predict

# Configuração
IMAGE_DIR = "../../flickr30k-images"
MODELS_DIR = "../models"

# Carregamento de dados e modelos
print("Carregando modelos e dados...")
model = SentenceTransformer(f"{MODELS_DIR}/model_sentence_transformer")
scaler = joblib.load(f"{MODELS_DIR}/model_scaler.joblib")
pca = joblib.load(f"{MODELS_DIR}/model_pca.joblib")
pca_hdbscan = joblib.load(f"{MODELS_DIR}/model_pca_hdbscan.joblib")
df = pd.read_csv(f"{MODELS_DIR}/descricao_clusters.csv")
data = np.load(f"{MODELS_DIR}/embeddings_pca.npz")

# Carregar descrições amostradas do HDBSCAN
hdbscan_data = np.load(f"{MODELS_DIR}/embeddings_hdbscan.npz")
hdbscan_labels = hdbscan_data["labels"]
desc_sample = pd.read_csv(f"{MODELS_DIR}/descricao_hdbscan.csv")

# Adiciona a coluna 'hdbscan_label' ao desc_samples
desc_sample['hdbscan_label'] = hdbscan_labels

# Labels salvos no .npz
label_map = {
    "kmeans_100": "kmeans_100",
    "kmeans_150": "kmeans_150",
    "kmeans_200": "kmeans_200"
}

# Carregamento de modelos KMeans
kmeans_models = {
    name: joblib.load(f"{MODELS_DIR}/{name}.joblib")
    for name in ["kmeans_100", "kmeans_150", "kmeans_200"]
}
hdbscan_model = joblib.load(f"{MODELS_DIR}/hdbscan.joblib")

# Preenche colunas de cluster no DataFrame
for nome_modelo, nome_npz in label_map.items():
    col_name = f"{nome_modelo}_label"
    if col_name not in df.columns and nome_npz in data:
        df[col_name] = data[nome_npz]

# Classificação de uma nova descrição
def classificar_descricao(texto, modelo_nome):
    print(f"\n>> Usando modelo: {modelo_nome}")

    # Determina o cluster
    if modelo_nome.startswith("kmeans"):
        emb = model.encode([texto], show_progress_bar=True, convert_to_numpy=True)
        emb = normalize(emb, norm='l2')
        emb = scaler.transform(emb)
        emb_pca = pca.transform(emb)
        cluster_id = kmeans_models[modelo_nome].predict(emb_pca)[0]

        col_cluster = f"{modelo_nome}_label"
        cluster_df = df[df[col_cluster] == cluster_id]

        if cluster_df.empty:
            print("Nenhuma descrição encontrada no cluster.")
            return cluster_id, None, None
    
        descs = cluster_df['description'].tolist()
        embs = normalize(model.encode(descs, show_progress_bar=True, convert_to_numpy=True), norm='l2')
        embs = scaler.transform(embs)
        embs_pca = pca.transform(embs)
        sims = cosine_similarity(emb_pca, embs_pca)[0]
        idx = np.argmax(sims)

        return cluster_id, descs[idx], cluster_df.iloc[idx]['filename']
    
    elif modelo_nome == "hdbscan":
        emb = model.encode([texto], show_progress_bar=True, convert_to_numpy=True)
        emb = normalize(emb, norm='l2')
        emb = scaler.transform(emb)
        emb_pca = pca_hdbscan.transform(emb)
        cluster_id, _ = approximate_predict(hdbscan_model, emb_pca)
        cluster_id = cluster_id[0]
        if cluster_id == -1:
            print("Classificado como RUÍDO pelo HDBSCAN.")
            return -1, None, None
        
        # Usa labels já carregados para as amostras usadas no treino
        cluster_ids = desc_sample['hdbscan_label'].values

        # Filtrar dataframe e embeddings amostrados pelo cluster predito
        cluster_df = desc_sample[cluster_ids == cluster_id]
        if cluster_df.empty:
            print("Nenhuma descrição encontrada no cluster.")
            return cluster_id, None, None
        
        # Buscar a descrição mais similar dentro do cluster
        descs = cluster_df['description'].tolist()
        embs = normalize(model.encode(descs, show_progress_bar=True, convert_to_numpy=True), norm='l2')
        embs = scaler.transform(embs)
        embs_pca = pca_hdbscan.transform(embs)
        sims = cosine_similarity(emb_pca, embs_pca)[0]
        idx = np.argmax(sims)

        return cluster_id, descs[idx], cluster_df.iloc[idx]['filename']
    
    else:
        raise ValueError("Modelo não reconhecido.")

# Execução
entrada = input("Digite uma descrição: ").strip()
modelos_disponiveis = list(kmeans_models.keys()) + ["hdbscan"]
print("Modelos disponíveis:", modelos_disponiveis)

modelo = input("Escolha o modelo (ex: kmeans_100, hdbscan): ").strip()
if modelo not in modelos_disponiveis:
    print("Modelo inválido.")
    exit()

cluster, similar_text, image = classificar_descricao(entrada, modelo)

# Resultado final
if cluster == -1 or image is None:
    print("Não foi possível encontrar uma imagem correspondente.")
else:
    print("\nResultado:")
    print(f"Cluster atribuído: {cluster}")
    print(f"Descrição mais similar: {similar_text}")
    print(f"Imagem associada: {image}")

    path = os.path.join(IMAGE_DIR, image)
    if os.path.exists(path):
        img = Image.open(path)
        plt.imshow(img)
        plt.title(f"Imagem do cluster {cluster}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Imagem '{path}' não encontrada.")

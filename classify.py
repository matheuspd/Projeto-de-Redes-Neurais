import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from PIL import Image
import os

#  CONFIGURAÇÃO 
IMAGE_DIR = "flickr30k-images"  # ajuste conforme necessário
CSV_PATH = "descricao_clusters.csv"

#  1. Carrega modelos e dados 
print("Carregando modelos e dados...")
model = SentenceTransformer("model_sentence_transformer")
pca = joblib.load("model_pca.joblib")
kmeans = joblib.load("model_kmeans.joblib")
df = pd.read_csv("descricao_clusters.csv")

#  2. Função para classificar nova descrição 
def classificar_nova_descricao(texto):
    emb = model.encode([texto], convert_to_numpy=True)
    emb_norm = normalize(emb, norm='l2')
    emb_pca = pca.transform(emb_norm)

    cluster_id = kmeans.predict(emb_pca)[0]
    print(f"Cluster previsto: {cluster_id}")

    # Filtra descrições no mesmo cluster
    cluster_df = df[df['cluster'] == cluster_id]
    
    # Recalcula os embeddings PCA do cluster
    cluster_embeddings = model.encode(cluster_df['description'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    cluster_embeddings = normalize(cluster_embeddings, norm='l2')
    cluster_embeddings = pca.transform(cluster_embeddings)

    # Similaridade com cada descrição do cluster
    sims = cosine_similarity(emb_pca, cluster_embeddings)[0]
    idx_mais_similar = np.argmax(sims)

    descricao_similar = cluster_df.iloc[idx_mais_similar]['description']
    imagem_similar = cluster_df.iloc[idx_mais_similar]['filename']
    
    return cluster_id, descricao_similar, imagem_similar

#  3. Exemplo de uso 
entrada = input("Digite uma descrição: ")
cluster, similar_text, image = classificar_nova_descricao(entrada)

print("\n Resultado ")
print(f"Cluster atribuído: {cluster}")
print(f"Descrição mais similar: {similar_text}")
print(f"Imagem associada: {image}")

#  4. Plot da imagem 
caminho_imagem = os.path.join(IMAGE_DIR, image)
if os.path.exists(caminho_imagem):
    print("Exibindo imagem associada...")
    img = Image.open(caminho_imagem)
    plt.imshow(img)
    plt.title(f"Imagem do cluster {cluster}")
    plt.axis('off')
    plt.show()
else:
    print(f"Imagem '{caminho_imagem}' não encontrada.")
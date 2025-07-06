import os
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt

# Caminhos
OUTPUT_FOLDER = "../clip_outputs"
CSV_PATH = "../../flickr_annotations_30k.csv"
IMG_FOLDER = "../../flickr30k-images/"

# Carrega modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Carrega embeddings de treino
data = np.load(os.path.join(OUTPUT_FOLDER, "clip_embeddings_pca.npz"), allow_pickle=True)
train_image_embeddings = data["embeddings"]  # já são embeddings combinados e reduzidos por PCA (usando KMeans)

# Carrega scaler e PCA
scaler = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_scaler.joblib"))
pca = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_pca.joblib"))

# Carrega nomes e descrições
filenames = pd.read_csv(os.path.join(OUTPUT_FOLDER, "clip_train_filenames.csv"))["filename"].tolist()
df_full = pd.read_csv(CSV_PATH)
df_full["raw"] = df_full["raw"].apply(eval)
df_full["description_concat"] = df_full["raw"].apply(lambda lst: " ".join(lst))
df_full = df_full[df_full["filename"].isin(filenames)]
df_full = df_full.set_index("filename")

# ========= Função de recuperação =========
def recuperar_imagens_por_texto(query_text, top_k=3):
    # Embedding do texto de consulta
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs).cpu().numpy()

    # Normaliza e aplica PCA
    combined = np.concatenate([np.zeros_like(text_emb), text_emb], axis=1)  # zeros para "imagem"
    scaled = scaler.transform(combined)
    query_embedding = pca.transform(scaled)

    # Similaridade com embeddings de imagem
    sim_scores = cosine_similarity(query_embedding, train_image_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    # Retorna resultados
    resultados = []
    for idx in top_indices:
        fname = filenames[idx]
        desc = df_full.loc[fname]["description_concat"]
        sim = sim_scores[idx]
        resultados.append((fname, desc, sim))

    return resultados

# ========= Função para plotar imagens recuperadas =========
def plot_imagens_recuperadas(resultados):
    plt.figure(figsize=(18, 3 * len(resultados)))
    for i, (fname, desc, sim) in enumerate(resultados):
        img_path = os.path.join(IMG_FOLDER, fname)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")

        plt.subplot(len(resultados), 1, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Arquivo: {fname} | Similaridade: {sim:.4f}\nDescrição: {desc[:250]}...", fontsize=8)

    plt.tight_layout()
    plt.show()

# ========= Exemplo de uso =========
consulta = input("\nDigite a descrição para busca de imagens: ").strip()

if consulta == "":
    print("Nenhuma descrição fornecida. Encerrando.")

else:
    resultados = recuperar_imagens_por_texto(consulta, top_k=3)

    for fname, desc, sim in resultados:
        print(f"Imagem: {fname} | Similaridade: {sim:.4f}")
        print(f"Descrição: {desc[:200]}...\n")

    plot_imagens_recuperadas(resultados)

"""
Script para extração de descrições visuais e textuais do dataset Flickr30k, geração de embeddings com CLIP,
redução de dimensionalidade com PCA, clustering com KMeans e HDBSCAN para comparação, e visualização dos clusters
com t-SNE. Os resultados são salvos para uso posterior.

Dataset: Flickr30k
------------------
O Flickr30k é um conjunto de dados com aproximadamente 30.000 imagens de cenas cotidianas, cada uma anotada com 5
descrições textuais (legendas) escritas por humanos. É amplamente utilizado em tarefas de visão computacional
e PLN, como busca cruzada imagem-texto e geração de legendas automáticas.

Observações:
---------------------
- Dado o tempo de processamento elevado, as 5 descrições são tratadas como uma única descrição para cada imagem.
- Foi aplicada a separação treino e teste baseado nas anotações do flickr_annotations_30k.csv para realizar 
testes de recuperação posteriormente.
- Só foi aplicado um modelo de KMeans e de HDBSCAN para comparação, diferentemente da parte 1, somente para
facilitar a implementação.

Etapas do pipeline:
---------------------
- Leitura das anotações do Flickr30k, usando apenas exemplos marcados como "train".
- Concatenação das 5 descrições de cada imagem em uma única string.
- Extração de embeddings de imagem e texto usando CLIP.
   - Embeddings são salvos em checkpoints parciais a cada 1024 imagens.
   - O script continua do último ponto automaticamente.
- Combinação, normalização e aplicação de PCA para reduzir dimensionalidade.
- Clusterização com KMeans e HDBSCAN.
- Visualização dos clusters com t-SNE.
- Salva os resultados e modelos.
"""

import os
import joblib
import warnings
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.cluster import KMeans
from openTSNE import TSNE
from sklearn.decomposition import PCA
import hdbscan
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ===================== Inicialização =====================
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Usando dispositivo: {device}")

IMG_FOLDER = "../../flickr30k-images/"
CSV_PATH = "../../flickr_annotations_30k.csv"
OUTPUT_FOLDER = "../clip_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== Carregamento do modelo CLIP =====================
print("[INFO] Carregando modelo CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# ===================== Carregamento do dataset =====================
print("[INFO] Carregando anotações...")
df = pd.read_csv(CSV_PATH)
df["raw"] = df["raw"].apply(literal_eval)
df["description_concat"] = df["raw"].apply(lambda lst: " ".join(lst))
df = df[df["split"] == "train"].reset_index(drop=True)
print(f"[INFO] Número de imagens de treino: {len(df)}")

# ===================== Função de extração de embeddings =====================
def extract_clip_embeddings(df, checkpoint_folder, batch_size=64, checkpoint_interval=1024):
    os.makedirs(checkpoint_folder, exist_ok=True)

    final_path = os.path.join(checkpoint_folder, "clip_embeddings_final.npz")
    if os.path.exists(final_path):
        print("[INFO] Embeddings já foram extraídos anteriormente. Carregando...")
        data = np.load(final_path, allow_pickle=True)
        return {
            "image_embeddings": data["image_embeddings"],
            "text_embeddings": data["text_embeddings"],
            "filenames": data["filenames"].tolist()
        }

    all_image_embeddings, all_text_embeddings, all_filenames = [], [], []

    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_folder) if f.startswith("clip_embeddings_part_")],
        key=lambda x: int(x.split("_")[3].split(".")[0])
    )

    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        print(f"[INFO] Carregando último checkpoint: {last_checkpoint}...")
        data = np.load(os.path.join(checkpoint_folder, last_checkpoint), allow_pickle=True)
        all_image_embeddings = list(data["image_embeddings"])
        all_text_embeddings = list(data["text_embeddings"])
        all_filenames = list(data["filenames"])

    filenames_set = set(all_filenames)
    df_remaining = df[~df['filename'].isin(filenames_set)].reset_index(drop=True)
    print(f"[INFO] Imagens restantes para processar: {len(df_remaining)}")

    batch_images, batch_texts, batch_filenames = [], [], []
    images_since_last_checkpoint = 0

    for row in tqdm(df_remaining.itertuples(), total=len(df_remaining), desc="Extraindo CLIP embeddings"):
        image_path = os.path.join(IMG_FOLDER, row.filename)
        if not os.path.exists(image_path):
            print(f"[WARN] Imagem não encontrada: {row.filename}, pulando.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            batch_images.append(image)
            batch_texts.append(row.description_concat)
            batch_filenames.append(row.filename)

            if len(batch_images) == batch_size:
                inputs = clip_processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    all_image_embeddings.extend(outputs.image_embeds.cpu().numpy())
                    all_text_embeddings.extend(outputs.text_embeds.cpu().numpy())
                    all_filenames.extend(batch_filenames)

                images_since_last_checkpoint += len(batch_filenames)
                batch_images, batch_texts, batch_filenames = [], [], []

                if images_since_last_checkpoint >= checkpoint_interval:
                    part_idx = len(all_filenames)
                    np.savez(os.path.join(checkpoint_folder, f"clip_embeddings_part_{part_idx}.npz"),
                             image_embeddings=np.array(all_image_embeddings),
                             text_embeddings=np.array(all_text_embeddings),
                             filenames=np.array(all_filenames))
                    print(f"[INFO] Checkpoint salvo com {part_idx} imagens.")
                    images_since_last_checkpoint = 0

        except Exception as e:
            print(f"[ERROR] Erro ao processar {row.filename}: {e}")

    if batch_images:
        inputs = clip_processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            all_image_embeddings.extend(outputs.image_embeds.cpu().numpy())
            all_text_embeddings.extend(outputs.text_embeds.cpu().numpy())
            all_filenames.extend(batch_filenames)

    np.savez(os.path.join(checkpoint_folder, "clip_embeddings_final.npz"),
             image_embeddings=np.array(all_image_embeddings),
             text_embeddings=np.array(all_text_embeddings),
             filenames=np.array(all_filenames))
    print("[INFO] Embeddings finais salvos.")

    return {
        "image_embeddings": np.array(all_image_embeddings),
        "text_embeddings": np.array(all_text_embeddings),
        "filenames": all_filenames
    }

print("[INFO] Iniciando extração de embeddings com CLIP...")
checkpoint_folder = os.path.join(OUTPUT_FOLDER, "checkpoints")
emb_dict = extract_clip_embeddings(df, checkpoint_folder)

# ===================== Processamento de Embeddings =====================
embed_df = pd.DataFrame({"filename": emb_dict["filenames"]})
embed_df.to_csv(os.path.join(OUTPUT_FOLDER, "clip_train_filenames.csv"), index=False)

print("[INFO] Concatenando embeddings...")
combined_embeddings = np.concatenate([emb_dict["image_embeddings"], emb_dict["text_embeddings"]], axis=1)

print("[INFO] Normalizando e aplicando PCA...")
scaler = StandardScaler()
scaled = scaler.fit_transform(combined_embeddings)
joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "clip_scaler.joblib"))

pca = PCA(n_components=scaled.shape[1] // 3, random_state=1337)
pca_embeddings = pca.fit_transform(scaled)
joblib.dump(pca, os.path.join(OUTPUT_FOLDER, "clip_pca.joblib"))

# ===================== KMeans =====================
print("[INFO] Aplicando KMeans...")
kmeans = KMeans(n_clusters=150, random_state=1337)
kmeans_labels = kmeans.fit_predict(pca_embeddings)
joblib.dump(kmeans, os.path.join(OUTPUT_FOLDER, "clip_kmeans.joblib"))

# ===================== HDBSCAN =====================
print("[INFO] Aplicando HDBSCAN...")
pca_hdbscan = PCA(n_components=60, random_state=1337)
pca_embeddings_hdbscan = pca_hdbscan.fit_transform(scaled)
joblib.dump(pca_hdbscan, os.path.join(OUTPUT_FOLDER, "clip_pca_hdbscan.joblib"))

hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=3, cluster_selection_epsilon=0.5, prediction_data=True)
hdb_labels = hdb.fit_predict(pca_embeddings_hdbscan)
joblib.dump(hdb, os.path.join(OUTPUT_FOLDER, "clip_hdbscan.joblib"))

# ===================== Visualização com t-SNE =====================
def plot_tsne(embeddings_2d, labels, model_name):
    df_plot = embed_df.copy()
    df_plot["x"] = embeddings_2d[:, 0]
    df_plot["y"] = embeddings_2d[:, 1]
    df_plot["cluster"] = labels
    palette = cc.glasbey[:len(set(labels))]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_plot, x="x", y="y", hue="cluster", palette=palette, legend=None, alpha=0.6)
    plt.title(f"t-SNE - {model_name}")
    plt.savefig(f"{OUTPUT_FOLDER}/{model_name}_tSNE.png", dpi=300)
    plt.close()
    print(f"[INFO] t-SNE salvo para {model_name}")

print("[INFO] Gerando visualizações t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_jobs=-1, random_state=1337)
tsne_emb = tsne.fit(pca_embeddings)
plot_tsne(tsne_emb, kmeans_labels, "KMeans")
plot_tsne(tsne_emb, hdb_labels, "HDBSCAN")

# ===================== Salvamento Final =====================
print("[INFO] Salvando embeddings finais...")
np.savez(os.path.join(OUTPUT_FOLDER, "clip_embeddings_pca.npz"),
         embeddings=pca_embeddings,
         embeddings_hdbscan=pca_embeddings_hdbscan,
         kmeans=kmeans_labels,
         hdbscan=hdb_labels)
print("[INFO] Processo concluído com sucesso.")

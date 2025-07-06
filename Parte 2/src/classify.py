import os
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import hdbscan
import matplotlib.pyplot as plt

# ============ Caminhos ============
IMG_FOLDER = "../../flickr30k-images/"
CSV_PATH = "../../flickr_annotations_30k.csv"
OUTPUT_FOLDER = "../clip_outputs"
TEST_EMB_PATH = os.path.join(OUTPUT_FOLDER, "clip_test_embeddings.npz")

# ============ Inicialização ============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Usando {device}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# ============ Carrega dados de teste ============
df = pd.read_csv(CSV_PATH)
df["raw"] = df["raw"].apply(literal_eval)
df["description_concat"] = df["raw"].apply(lambda lst: " ".join(lst))
df_test = df[df["split"] == "test"].reset_index(drop=True)
print(f"[INFO] Total de imagens de teste: {len(df_test)}")

# ============ Carrega modelos ============
scaler = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_scaler.joblib"))
pca_kmeans = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_pca.joblib"))
pca_hdbscan = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_pca_hdbscan.joblib"))
kmeans = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_kmeans.joblib"))
hdb = joblib.load(os.path.join(OUTPUT_FOLDER, "clip_hdbscan.joblib"))

# ============ Carrega embeddings de treino ============
train_data = np.load(os.path.join(OUTPUT_FOLDER, "clip_embeddings_pca.npz"), allow_pickle=True)
train_embeddings_kmeans = train_data["embeddings"]
train_embeddings_hdbscan = train_data["embeddings_hdbscan"]
train_filenames = pd.read_csv(os.path.join(OUTPUT_FOLDER, "clip_train_filenames.csv"))["filename"].tolist()

# ============ Processa embeddings das imagens de teste ============
if os.path.exists(TEST_EMB_PATH):
    print("[INFO] Carregando embeddings de teste existentes...")
    loaded = np.load(TEST_EMB_PATH)
    test_image_embeddings = loaded["image_embeds"]
    test_text_embeddings = loaded["text_embeds"]
    test_filenames = loaded["filenames"].tolist()
else:
    test_image_embeddings = []
    test_text_embeddings = []
    test_filenames = []
    print("[INFO] Extraindo embeddings CLIP das imagens de teste...")
    for row in tqdm(df_test.itertuples(), total=len(df_test)):
        image_path = os.path.join(IMG_FOLDER, row.filename)
        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(text=row.description_concat, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
            test_image_embeddings.append(outputs.image_embeds.cpu().numpy()[0])
            test_text_embeddings.append(outputs.text_embeds.cpu().numpy()[0])
            test_filenames.append(row.filename)
        except Exception as e:
            print(f"[ERRO] {row.filename}: {e}")

    # Salva embeddings e filenames em arquivo npz
    np.savez(TEST_EMB_PATH,
             image_embeds=np.array(test_image_embeddings),
             text_embeds=np.array(test_text_embeddings),
             filenames=np.array(test_filenames))
    print(f"[INFO] Embeddings de teste salvos em {TEST_EMB_PATH}")

# ============ Combina, normaliza, projeta ============
print("[INFO] Aplicando scaler e PCA nos dados de teste...")

test_embeddings = np.concatenate([test_image_embeddings, test_text_embeddings], axis=1)
scaled = scaler.transform(test_embeddings)

pca_test_kmeans = pca_kmeans.transform(scaled)
pca_test_hdbscan = pca_hdbscan.transform(scaled)

# ============ Atribui clusters ============
kmeans_preds = kmeans.predict(pca_test_kmeans)
hdbscan_preds, _ = hdbscan.approximate_predict(hdb, pca_test_hdbscan)

# ============ Pré-processa clusters ============
train_labels_hdb = hdb.labels_
cluster_to_indices_kmeans = {}
cluster_to_indices_hdb = {}

for i, label in enumerate(kmeans.labels_):
    cluster_to_indices_kmeans.setdefault(label, []).append(i)

for i, label in enumerate(train_labels_hdb):
    cluster_to_indices_hdb.setdefault(label, []).append(i)

# ============ Calcula similaridades ============
results = []
for i, filename in enumerate(test_filenames):
    emb_k = pca_test_kmeans[i].reshape(1, -1)
    emb_h = pca_test_hdbscan[i].reshape(1, -1)

    # KMeans
    cluster_k = kmeans_preds[i]
    idxs_k = cluster_to_indices_kmeans.get(cluster_k, [])
    sim_k = cosine_similarity(emb_k, train_embeddings_kmeans[idxs_k])[0]
    best_k_idx = idxs_k[np.argmax(sim_k)]
    best_k_filename = train_filenames[best_k_idx]
    best_k_score = sim_k[np.argmax(sim_k)]

    # HDBSCAN
    cluster_h = hdbscan_preds[i]
    best_h_filename = None
    best_h_score = None
    if cluster_h != -1 and cluster_h in cluster_to_indices_hdb:
        idxs_h = cluster_to_indices_hdb[cluster_h]
        sim_h = cosine_similarity(emb_h, train_embeddings_hdbscan[idxs_h])[0]
        best_h_idx = idxs_h[np.argmax(sim_h)]
        best_h_filename = train_filenames[best_h_idx]
        best_h_score = sim_h[np.argmax(sim_h)]

    results.append({
        "filename_teste": filename,
        "cluster_kmeans": int(cluster_k),
        "cluster_hdbscan": int(cluster_h),
        "mais_proxima_kmeans": best_k_filename,
        "similaridade_kmeans": round(float(best_k_score), 4),
        "mais_proxima_hdbscan": best_h_filename,
        "similaridade_hdbscan": round(float(best_h_score), 4) if best_h_score else None,
    })

# ============ Exibe resultados ============
print("\nExemplos de correspondência (primeiros 10):\n")
for r in results[:10]:
    print(f"[{r['filename_teste']}]")
    print(f"  * KMeans    => Cluster #{r['cluster_kmeans']} | Mais próxima: {r['mais_proxima_kmeans']} (sim={r['similaridade_kmeans']})")
    if r['cluster_hdbscan'] == -1 or r['mais_proxima_hdbscan'] is None:
        print(f"  * HDBSCAN   => Ruído (sem cluster atribuído)")
    else:
        print(f"  * HDBSCAN   => Cluster #{r['cluster_hdbscan']} | Mais próxima: {r['mais_proxima_hdbscan']} (sim={r['similaridade_hdbscan']})")
    print()

# ============ Carrega descrições ============
df_all = pd.read_csv(CSV_PATH)
df_all["raw"] = df_all["raw"].apply(literal_eval)
desc_map = dict(zip(df_all["filename"], df_all["raw"]))

# ============ Visualização ============
def show_image_with_desc(ax, path, desc_list, title):
    try:
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        desc = "\n".join(["* " + d for d in desc_list])
        ax.text(0.5, -0.15, desc, ha='center', va='top', wrap=True, fontsize=6, transform=ax.transAxes)
    except:
        ax.set_title("Erro ao carregar imagem")
        ax.axis("off")

print("\n[INFO] Exibindo comparações com imagens similares por modelo...\n")

for i, r in enumerate(results[:5]):
    filename_test = r["filename_teste"]
    desc_test = desc_map.get(filename_test, ["(descrição não encontrada)"])
    desc_k = desc_map.get(r["mais_proxima_kmeans"], ["(descrição não encontrada)"])
    desc_h = desc_map.get(r["mais_proxima_hdbscan"], ["[RUÍDO] Nenhum cluster atribuído"]) if r["mais_proxima_hdbscan"] else ["[RUÍDO] Nenhum cluster atribuído"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    show_image_with_desc(axes[0], os.path.join(IMG_FOLDER, filename_test), desc_test, f"[Teste] {filename_test}")
    show_image_with_desc(axes[1], os.path.join(IMG_FOLDER, r["mais_proxima_kmeans"]), desc_k, f"[KMeans] {r['mais_proxima_kmeans']}\n(sim={r['similaridade_kmeans']:.4f})")
    if r["mais_proxima_hdbscan"]:
        show_image_with_desc(axes[2], os.path.join(IMG_FOLDER, r["mais_proxima_hdbscan"]), desc_h, f"[HDBSCAN] {r['mais_proxima_hdbscan']}\n(sim={r['similaridade_hdbscan']:.4f})")
    else:
        axes[2].set_title("[HDBSCAN] Ruído", fontsize=10)
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Sem cluster atribuído", ha='center', va='center')

    plt.suptitle(f"Clusters para {filename_test} - KMeans #{r['cluster_kmeans']}, HDBSCAN #{r['cluster_hdbscan']}", fontsize=12)
    plt.tight_layout()
    plt.show()

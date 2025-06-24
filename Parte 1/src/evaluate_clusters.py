import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample

MODELS_DIR = "../models"
CSV_PATH = f"{MODELS_DIR}/descricao_clusters.csv"
CSV_HDBSCAN_PATH = f"{MODELS_DIR}/descricao_hdbscan.csv"

print("Carregando embeddings e labels...")

# Dados PCA e labels dos kmeans
data = np.load(f"{MODELS_DIR}/embeddings_pca.npz")
embeddings = data['embeddings']
df = pd.read_csv(CSV_PATH)

# Dados amostrados e labels do HDBSCAN
desc_sample = pd.read_csv(CSV_HDBSCAN_PATH)
hdbscan_data = np.load(f"{MODELS_DIR}/embeddings_hdbscan.npz")
embeddings_hdb = hdbscan_data["embeddings"]
hdbscan_labels = hdbscan_data["labels"]

desc_sample["hdbscan_label"] = hdbscan_labels

# Modelos e nomes das colunas de label no dataframe
modelos = {
    "kmeans_100": "kmeans_100_label",
    "kmeans_150": "kmeans_150_label",
    "kmeans_200": "kmeans_200_label",
    "hdbscan": "hdbscan_label"
}

# Adiciona colunas de labels no DataFrame a partir do arquivo .npz
for modelo, col_label in modelos.items():
    nome_npz = modelo  # chave no .npz suposta igual ao nome do modelo
    if col_label not in df.columns:
        if nome_npz in data:
            df[col_label] = data[nome_npz]
        else:
            print(f"[AVISO] Chave '{nome_npz}' não encontrada no arquivo .npz; coluna '{col_label}' não adicionada.")

sample_size = 20000

print("\n=== Avaliação dos Modelos ===\n")
for nome, col_label in modelos.items():
    print(f">>> Avaliando modelo: {nome}")
    
    if nome == "hdbscan":
        # Para HDBSCAN, usa embeddings e labels amostrados
        if col_label not in desc_sample.columns:
            print(f"  [ERRO] Coluna {col_label} não encontrada no CSV de amostra.")
            continue
        
        labels = desc_sample[col_label].values
        mask_validos = labels != -1  # remove ruído
        X = embeddings_hdb[mask_validos]
        y = labels[mask_validos]

    else:
        # Para KMeans, usa df completo e embeddings PCA
        if col_label not in df.columns:
            print(f"  [ERRO] Coluna {col_label} não encontrada no CSV principal.")
            continue
        
        labels = df[col_label].values
        mask_validos = labels != -1
        X = embeddings[mask_validos]
        y = labels[mask_validos]

    n_clusters = len(set(y))
    if -1 in y:
        n_clusters -= 1

    if n_clusters < 2:
        print("  [AVISO] Clusters insuficientes para avaliação.")
        continue

    print(f"  Clusters válidos: {n_clusters}")
    
    # Silhouette Score com amostragem
    try:
        X_sample, y_sample = resample(
            X, y,
            n_samples=min(sample_size, len(X)),
            stratify=y,
            random_state=1337
        )
        silh = silhouette_score(X_sample, y_sample)
        print(f"  Silhouette Score (amostrado): {silh:.4f}")
    except Exception as e:
        print(f"  [ERRO] Silhouette Score: {e}")

    # Davies-Bouldin Index
    try:
        db = davies_bouldin_score(X, y)
        print(f"  Davies-Bouldin Index: {db:.4f}")
    except Exception as e:
        print(f"  [ERRO] Davies-Bouldin Index: {e}")

    # Calinski-Harabasz Index
    try:
        ch = calinski_harabasz_score(X, y)
        print(f"  Calinski-Harabasz Index: {ch:.4f}")
    except Exception as e:
        print(f"  [ERRO] Calinski-Harabasz Index: {e}")

# Silhouette Score
#   Mede o quão similar um ponto é ao seu próprio cluster comparado aos outros.
#   Varia de -1 a 1. Valores mais altos indicam clusters mais definidos.
# Demora muito para rodar, tiramos uma amostra por causa disso.
# Amostra (ex: 20 mil pontos aleatórios com mesma distribuição de clusters)

# Davies-Bouldin Index
#   Mede a média da razão entre a distância intra-cluster e inter-cluster.
#   Valores menores indicam melhores clusters.

# Calinski-Harabasz Index
#   Razão entre dispersão entre clusters e dentro dos clusters.
#   Quanto maior, melhor.

"""
Exemplo resultados obtidos anteriormente:
Avaliando modelo: kmeans_100
  Clusters válidos: 100
  Silhouette Score (amostrado): 0.0496
  Davies-Bouldin Index: 3.4308
  Calinski-Harabasz Index: 832.8116

Avaliando modelo: kmeans_150   
  Clusters válidos: 150
  Silhouette Score (amostrado): 0.0501
  Davies-Bouldin Index: 3.3517
  Calinski-Harabasz Index: 623.3065

Avaliando modelo: kmeans_200
  Clusters válidos: 200
  Silhouette Score (amostrado): 0.0483
  Davies-Bouldin Index: 3.3196
  Calinski-Harabasz Index: 507.5002

Avaliando modelo: hdbscan
  Clusters válidos: 129
  Silhouette Score (amostrado): 0.1717
  Davies-Bouldin Index: 1.5490
  Calinski-Harabasz Index: 168.2443
"""

# O valor do Silhouette Score é bem baixo, indicando que as descrições não estão claramente agrupadas, ou que os clusters se **sobrepõem bastante**.
# O valor do Davies-Bouldin Index reforça que há muita sobreposição entre clusters.
# O valor do Calinski-Harabasz Index parece razoável, mas não compensa as outras duas métricas.
# Falar sobre a diferença pro HDBSCAN mas como ele é ruim porque discarta 90% dos dados como ruído.

# Pela própria visualização dos cluster é possível perceber a grande sobreposição deles.

# Avaliar com divisão de treino e teste na parte 2 com CLIP (embedding imagem + texto).
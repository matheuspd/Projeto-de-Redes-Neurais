import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample

print("Carregando embeddings e labels...")
data = np.load("embeddings_labels.npz")
embeddings = data['embeddings']
labels = data['labels']

print("Calculando métricas de avaliação...")

# Silhouette Score
#   Mede o quão similar um ponto é ao seu próprio cluster comparado aos outros.
#   Varia de -1 a 1. Valores mais altos indicam clusters mais definidos.
# Demora muito para rodar, tiramos uma amostra por causa disso.
# Amostra (ex: 20 mil pontos aleatórios com mesma distribuição de clusters)
sample_size = 20000
embeddings_sample, labels_sample = resample(
    embeddings, labels,
    n_samples=sample_size,
    stratify=labels,
    random_state=1337
)

silh_score = silhouette_score(embeddings_sample, labels_sample)
print(f"Silhouette Score (amostrado): {silh_score:.4f}")

# Davies-Bouldin Index
#   Mede a média da razão entre a distância intra-cluster e inter-cluster.
#   Valores menores indicam melhores clusters.
db = davies_bouldin_score(embeddings, labels)
print(f"Davies-Bouldin Index: {db:.4f}")

# Calinski-Harabasz Index
#   Razão entre dispersão entre clusters e dentro dos clusters.
#   Quanto maior, melhor.
ch = calinski_harabasz_score(embeddings, labels)
print(f"Calinski-Harabasz Index: {ch:.4f}")

# Resultados último modelo:
# Silhouette Score (amostrado): 0.0460
# Davies-Bouldin Index: 3.5088
# Calinski-Harabasz Index: 632.4417

# O valor do Silhouette Score é bem baixo, indicando que as descrições não estão claramente agrupadas, ou que os clusters se **sobrepõem bastante**.
# O valor do Davies-Bouldin Index reforça que há muita sobreposição entre clusters.
# O valor do Calinski-Harabasz Index parece razoável, mas não compensa as outras duas métricas.

# Pela própria visualização dos cluster é possível perceber a grande sobreposição deles.

# Avaliar com divisão de treino e teste na parte 2 com CLIP (embedding imagem + texto).
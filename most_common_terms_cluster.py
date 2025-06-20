import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = list(stopwords.words("english"))

# Carrega o CSV
df = pd.read_csv("descricao_clusters.csv")

# Agrupa descrições por cluster
cluster_groups = df.groupby("cluster")["description"].apply(list)

# Prepara texto por cluster
cluster_docs = {
    cluster: " ".join(descriptions)
    for cluster, descriptions in cluster_groups.items()
}

# Cria vetor TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
tfidf_matrix = vectorizer.fit_transform(cluster_docs.values())
feature_names = vectorizer.get_feature_names_out()

# Exibe top termos por cluster
top_k = 10
print("\n Top palavras por cluster (TF-IDF) ")
for i, (cluster_id, doc) in enumerate(cluster_docs.items()):
    tfidf_scores = tfidf_matrix[i].toarray().flatten()
    top_indices = tfidf_scores.argsort()[::-1][:top_k]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"Cluster {cluster_id}: {', '.join(top_words)}")
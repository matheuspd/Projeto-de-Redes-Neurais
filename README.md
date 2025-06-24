# Projeto-de-Redes-Neurais
Projeto para a disciplina de redes neurais. Utilização de modelos de classificação e agrupamento de imagens e texto descritivo para análise e recuperação de conteúdo.

## Configuração do Ambiente

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd Projeto-de-Redes-Neurais
```

### 2. Execute o script de configuração
```bash
bash setup.sh
```

Este script irá:
- Baixar automaticamente o dataset Flickr30k do Hugging Face
- Criar um ambiente virtual Python
- Instalar todas as dependências necessárias

### 3. Ative o ambiente virtual
```bash
source venv/bin/activate
```

## Como Executar

### 1. Criar o Modelo (`create_model.py`)
```bash
python create_model.py
```

**O que faz:**
- Carrega o dataset Flickr30k com 5 descrições por imagem
- Gera embeddings textuais usando SentenceTransformer
- Aplica PCA para redução de dimensionalidade
- Realiza clusterização com KMeans (125 clusters)
- Aplica UMAP para visualização 2D
- Salva os modelos treinados e gera visualização dos clusters
- **Arquivos gerados:** `descricao_clusters.csv`, `model_kmeans.joblib`, `model_pca.joblib`, `model_sentence_transformer/`, `embeddings_labels.npz`, `clusters_visualizacao.png`

### 2. Avaliar Clusters (`evaluate_clusters.py`)
```bash
python evaluate_clusters.py
```

**O que faz:**
- Calcula 3 métricas de avaliação da clusterização:
  - **Silhouette Score:** Mede a qualidade dos clusters (-1 a 1, maior é melhor)
  - **Davies-Bouldin Index:** Mede sobreposição entre clusters (menor é melhor)
  - **Calinski-Harabasz Index:** Razão dispersão inter/intra clusters (maior é melhor)
- Usa amostragem para acelerar o cálculo do Silhouette Score
- **Interpretação dos resultados:** 
  - Silhouette Score baixo (0.0460) indica clusters com muita sobreposição
  - Davies-Bouldin Index alto (3.5088) confirma a sobreposição entre clusters
  - Sugere que as descrições não estão claramente agrupadas
  - Visualização dos clusters também mostra grande sobreposição

### 3. Descrever Clusters (`describe_clusters.py`)
```bash
python describe_clusters.py
```

**O que faz:**
- Encontra uma descrição representativa para cada cluster
- Calcula o centroide de cada cluster no espaço PCA
- Seleciona a descrição mais próxima ao centroide usando similaridade de cosseno
- Permite escolher quantos clusters mostrar interativamente

### 4. Classificar Nova Descrição (`classify.py`)
```bash
python classify.py
```

**O que faz:**
- Recebe uma descrição textual como input do usuário
- Classifica a descrição em um dos clusters existentes
- Encontra a descrição mais similar dentro do cluster
- Retorna a imagem associada à descrição mais similar
- Exibe a imagem usando matplotlib
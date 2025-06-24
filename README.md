# Projeto-de-Redes-Neurais
Projeto para a disciplina de redes neurais. Utilização de modelos de classificação e agrupamento de imagens e texto descritivo para análise e recuperação de conteúdo.

# Configuração do Ambiente

## 1. Clone o repositório

```bash
git clone https://github.com/matheuspd/Projeto-de-Redes-Neurais.git
cd Projeto-de-Redes-Neurais
```

## 2. Execute o script de configuração

```bash
bash setup.sh
```

Este script irá:
- Baixar automaticamente o dataset Flickr30k do Hugging Face.
- Criar um ambiente virtual Python.
- Instalar todas as dependências necessárias nesse ambiente virtual.

## 3. Ative o ambiente virtual (caso não esteja ativado)

```bash
source venv/bin/activate
```

# Como Executar

## Parte 1:

Análise apenas textual.

```bash
cd Parte 1/src
```

### 1. Criar o Modelo (`create_model.py`)

```bash
python create_model.py
```

**O que faz:**
- Carrega o dataset Flickr30k contendo cerca de 30.000 imagens com 5 descrições cada.
- Expande as descrições para ter uma linha por descrição textual.
- Gera embeddings textuais usando o modelo pré-treinado SentenceTransformer (all-MiniLM-L6-v2).
- Normaliza os embeddings para norma L2.
- Aplica PCA para redução de dimensionalidade.
- Realiza clustering com três modelos KMeans (com 100, 150 e 200 clusters) e com um modelo HDBSCAN em uma amostra reduzida para maior eficiência.
- Gera visualizações dos clusters usando UMAP e t-SNE para os dois tipos de clusterização com amostras reduzidas para menor tempo de processamento.
- Salva os modelos treinados, embeddings, labels, e arquivos CSV com as descrições e seus clusters.
- Gera plots para análise visual dos clusters.
- **Arquivos gerados no diretório "Parte 1":** 
  - `models/descricao_clusters.csv`
  - `models/descricao_hdbscan.csv`
  - `models/kmeans_{num_clusters}.joblib`
  - `models/hdbscan.joblib`
  - `models/model_pca.joblib`
  - `models/model_pca_hdbscan.joblib`
  - `models/model_sentence_transformer/`
  - `models/embeddings_pca.npz`
  - `models/embeddings_hdbscan.npz`
  - `plots/HDBSCAN_{UMAP/tSNE}.png`
  - `plots/kmeans_{num_clusters}_{UMAP/tSNE}`

### 2. Avaliar Clusters (`evaluate_clusters.py`)

```bash
python evaluate_clusters.py
```

**O que faz:**
- Carrega embeddings e labels dos clusters para os modelos KMeans e HDBSCAN.
- Para cada modelo, avalia a qualidade dos clusters com três métricas padrão:
  - **Silhouette Score:** Avalia coesão e separação dos clusters (de -1 a 1, valores maiores indicam clusters mais bem definidos).
  - **Davies-Bouldin Index:** Mede a relação entre dispersão intra-cluster e separação inter-cluster (valores menores indicam clusters melhores).
  - **Calinski-Harabasz Index:** Relação entre dispersão entre clusters e dentro dos clusters (valores maiores indicam melhor separação).
- Usa amostragem para acelerar o cálculo do Silhouette Score.
- Trata clusters de ruído (-1) do HDBSCAN adequadamente.
- **Interpretação dos resultados:** 
  - Silhouette Score baixo indica clusters com muita sobreposição.
  - Davies-Bouldin Index alto confirma a sobreposição entre clusters.
  - Sugere que as descrições não estão claramente agrupadas.
  - Visualização dos clusters também mostra grande sobreposição.
  - HDBSCAN proporciona resultados melhores, porém considera aproximadamente 90% dos dados como ruído, tornando-o extremamente ruim em tarefas de recuperação de conteúdo para esse modelagem.

### 3. Descrever Clusters (`describe_clusters.py`)

```bash
python describe_clusters.py
```

**O que faz:**
- Carrega descrições, embeddings e labels de clusters para os modelos KMeans e HDBSCAN.
- Para cada cluster de um modelo escolhido pelo usuário:
  - Calcula o centroide dos embeddings das descrições dentro do cluster.
  - Identifica a descrição mais próxima ao centroide usando similaridade de cosseno.
  - Considera essa descrição como representativa daquele cluster.
- Permite escolher quantos clusters mostrar suas descrições representativas.

### 4. Classificar Nova Descrição (`classify.py`)

```bash
python classify.py
```

**O que faz:**
- Recebe uma descrição textual e o modelo de clusterização desejado como entrada do usuário.
- Gera o embedding da nova descrição com o modelo SentenceTransformer carregado.
- Aplica a mesma redução de dimensionalidade usada no treinamento (PCA ou PCA para HDBSCAN).
- Usa o modelo KMeans ou HDBSCAN para prever o cluster da nova descrição.
- Para o cluster previsto:
  - Encontra a descrição mais similar dentro do cluster.
  - Retorna o cluster atribuído, a descrição textual mais próxima, e o nome do arquivo da imagem associada.
- Exibe a imagem usando matplotlib.
- Informa se a descrição foi classificada como ruído (no caso do HDBSCAN).

### 5. Termos mais comuns nos cluster (`most_common_terms_cluster.py`)

```bash
python most_common_terms_cluster.py
```

**O que faz:**
- Analisa as descrições textuais agrupadas por cluster em cada modelo (KMeans e HDBSCAN).
- Aplica TF-IDF para extrair os termos mais relevantes (mais distintivos) em cada cluster, ignorando stopwords em inglês.
- Para o HDBSCAN, usa o conjunto amostrado de descrições (descricao_hdbscan.csv) e seus respectivos rótulos, que são os dados usados para treinar o modelo.
- Para os KMeans, usa as descrições e rótulos completos do dataset.
- Permite escolher quantos clusters mostrar e exibe os termos mais frequentes e importantes em cada um deles.
- Facilita a interpretação e o entendimento semântico dos clusters encontrados, mostrando palavras-chave típicas para cada grupo.

## Parte 2


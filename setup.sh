#!/bin/bash

# Script para baixar o dataset e configurar o projeto
# Executa: bash setup.sh

# 1. Baixar dataset do Hugging Face
echo "Baixando dataset Flickr30k..."
wget -O flickr30k-images.zip "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
echo "Descompactando imagens..."
unzip -q flickr30k-images.zip
rm flickr30k-images.zip
rm -rf __MACOSX

# 2. Criar ambiente virtual
echo "Criando ambiente virtual Python..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 3. Instalar dependências
echo "Instalando dependências..."
pip install -r requirements.txt

echo "=== Configuração concluída! ==="
echo ""
echo "Para ativar o ambiente virtual:"
echo "source venv/bin/activate"
echo ""
echo "Dataset baixado em: flickr30k-images/"

# Este exemplo demonstra como representar a mesma frase de duas formas diferentes (TF-IDF e Embedding) e como a similaridade semântica se comporta de maneira diferente da similaridade lexical.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Dados de Exemplo
consulta = ["ataque cardiaco"]
documentos = [
    "O paciente sofreu um infarto agudo do miocardio", # Doc1 (Relevante semanticamente)
    "A linguagem Python é ótima para ciencia de dados" # Doc2 (Irrelevante)
]
todos_textos = consulta + documentos

# --- PARTE 1: BUSCA POR PALAVRAS-CHAVE (TF-IDF) ---
print("="*50)
print("1. RESULTADOS DA BUSCA POR PALAVRAS-CHAVE (TF-IDF)")
print("="*50)

# Inicializa o vetorizador TF-IDF
# (não remove stopwords para simplificar, mas em geral, remove-se)
vectorizer_tfidf = TfidfVectorizer()

# Aplica o fit_transform em todos os textos e transforma consulta e docs
matriz_tfidf = vectorizer_tfidf.fit_transform(todos_textos)
vetor_consulta_tfidf = matriz_tfidf[0:1] # Primeiro elemento é a consulta
vetores_docs_tfidf = matriz_tfidf[1:] # Os demais são os documentos

# Calcula a similaridade de cosseno
similaridades_tfidf = cosine_similarity(vetor_consulta_tfidf, vetores_docs_tfidf).flatten()

# Exibe os resultados
print("Vocabulário (palavras únicas):", vectorizer_tfidf.get_feature_names_out())
for i, doc in enumerate(documentos):
    print(f"Similaridade entre '{consulta[0]}' e '{doc[:30]}...': {similaridades_tfidf[i]:.4f}")

# --- PARTE 2: BUSCA SEMÂNTICA (Sentence Transformers) ---
print("\n" + "="*50)
print("2. RESULTADOS DA BUSCA SEMÂNTICA (EMBEDDINGS)")
print("="*50)

# Carrega um modelo pré-treinado de embeddings para sentenças (em português)
# Vamos usar um modelo multilíngue que funciona bem para PT-BR
modelo_embedding = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Gera os embeddings para todos os textos
embeddings = modelo_embedding.encode(todos_textos)
embedding_consulta = embeddings[0].reshape(1, -1)
embeddings_docs = embeddings[1:]

# Calcula a similaridade de cosseno
similaridades_semanticas = cosine_similarity(embedding_consulta, embeddings_docs).flatten()

# Exibe os resultados
for i, doc in enumerate(documentos):
    print(f"Similaridade semântica entre '{consulta[0]}' e '{doc[:30]}...': {similaridades_semanticas[i]:.4f}")

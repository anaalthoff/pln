# Este exemplo demonstra:
# Cálculo de similaridade por cosseno entre pares de frases
# Comparação entre similaridade lexical (TF-IDF) e semântica (embeddings)
# Visualização da matriz de similaridade
# Identificação dos documentos mais similares a uma consulta

# ============================================================================
# Similaridade Semântica entre Frases e Documentos
# Exemplo Computacional Completo
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import seaborn as sns
import pandas as pd

# ============================================================================
# 1. DEFINIÇÃO DO CORPUS
# ============================================================================

# Frases em português para demonstrar similaridade semântica
frases = [
    # Frases sobre carros (semanticamente relacionadas)
    "O carro elétrico tem autonomia de 500 quilômetros",
    "O veículo movido a bateria pode percorrer 500 km",
    "O automóvel novo é muito rápido e econômico",
    "A empresa lançou um sedã elétrico com grande alcance",
    
    # Frases sobre receitas (tópico diferente)
    "Para fazer um bolo de chocolate, você precisa de farinha e ovos",
    "A receita de brownie leva cacau em pó e manteiga",
    
    # Frase de consulta (paciente com dor lombar - exemplo médico)
    "Paciente com dor lombar relata dificuldade para andar"
]

# Rótulos para identificar os grupos semanticamente relacionados
rotulos = [
    "Carro 1", "Carro 2", "Carro 3", "Carro 4",
    "Receita 1", "Receita 2",
    "Consulta"
]

print("=" * 70)
print("EXEMPLO COMPUTACIONAL: SIMILARIDADE SEMÂNTICA ENTRE FRASES")
print("=" * 70)

print(f"\nCorpus de {len(frases)} frases:")
for i, (rotulo, frase) in enumerate(zip(rotulos, frases)):
    print(f"  {rotulo:12}: {frase[:60]}...")

# ============================================================================
# 2. ABORDAGEM 1: SIMILARIDADE COM TF-IDF (LEXICAL)
# ============================================================================

print("\n" + "=" * 60)
print("2. SIMILARIDADE COM TF-IDF (Abordagem Lexical)")
print("=" * 60)

# Limpeza manual de stopwords em português
stopwords_pt = [
    'a', 'o', 'e', 'de', 'da', 'do', 'que', 'com', 'para', 'um', 'uma',
    'os', 'as', 'em', 'no', 'na', 'por', 'mais', 'menos', 'muito', 'pouco',
    'já', 'como', 'foi', 'ao', 'pela', 'pelo', 'sua', 'seu', 'seus', 'suas'
]

# Cria vetorizador TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords_pt)

# Calcula matriz TF-IDF
matriz_tfidf = vectorizer.fit_transform(frases)
vetores_tfidf = matriz_tfidf.toarray()

print(f"\nVocabulário de {len(vectorizer.get_feature_names_out())} termos")
print(f"Matriz TF-IDF: {matriz_tfidf.shape} (documentos × termos)")

# Calcula matriz de similaridade por cosseno
similaridade_tfidf = cosine_similarity(matriz_tfidf)


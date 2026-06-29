# ============================================================================
# Tópico Representação Semântica de Documentos para Busca
# Exemplo Computacional Completo
# ============================================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

# ============================================================================
# 1. DEFINIÇÃO DO CORPUS E CONSULTA
# ============================================================================

documentos = [
    "O carro elétrico tem autonomia de 500 quilômetros com uma única carga.",
    "O veículo movido a bateria oferece 500 km de alcance com uma carga completa.",
    "A receita de bolo de chocolate leva farinha, ovos e achocolatado em pó.",
    "O preço do automóvel elétrico caiu significativamente no último ano.",
    "O Nissan Leaf é um carro elétrico com boa autonomia para uso urbano.",
    "A indústria automotiva está investindo pesado em veículos elétricos.",
]

consulta = "automóvel elétrico autonomia"

print("=" * 70)
print("EXEMPLO COMPUTACIONAL: REPRESENTAÇÃO DE DOCUMENTOS PARA BUSCA")
print("=" * 70)
print(f"\nConsulta: '{consulta}'")
print(f"Número de documentos: {len(documentos)}\n")

# ============================================================================
# 2. DEFINIÇÃO DE STOP WORDS EM PORTUGUÊS
# ============================================================================

stopwords_pt = [
    'a', 'e', 'o', 'que', 'de', 'da', 'do', 'das', 'dos', 'em', 'com', 'para',
    'um', 'uma', 'os', 'as', 'seu', 'sua', 'seus', 'suas', 'na', 'no', 'nas',
    'nos', 'ao', 'aos', 'pela', 'pelo', 'pelas', 'pelos', 'este', 'esta', 'estes',
    'estas', 'isto', 'isso', 'aquilo', 'aquele', 'aquela', 'aqueles', 'aquelas',
    'me', 'te', 'se', 'nos', 'lhe', 'lhes', 'é', 'ser', 'ter', 'vir', 'está',
    'estão', 'estiver', 'foi', 'foram', 'será', 'serão', 'uma', 'com', 'de', 'o'
]

# ============================================================================
# 3. ABORDAGEM CLÁSSICA: TF-IDF (REPRESENTAÇÃO ESPARSA)
# ============================================================================

print("=" * 50)
print("3. ABORDAGEM CLÁSSICA - TF-IDF (Vetor Esparso)")
print("=" * 50)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=stopwords_pt,
    min_df=1,
    max_df=1.0,
    use_idf=True,
    smooth_idf=True,
    norm='l2'
)

start_time = time.time()
matriz_tfidf = vectorizer.fit_transform(documentos)
vetor_consulta_tfidf = vectorizer.transform([consulta])
tempo_tfidf = time.time() - start_time

similaridades_tfidf = cosine_similarity(vetor_consulta_tfidf, matriz_tfidf).flatten()

vocabulario = vectorizer.get_feature_names_out()
print(f"\nTamanho do vocabulário: {len(vocabulario)} termos")
print(f"Dimensionalidade do vetor TF-IDF: {matriz_tfidf.shape[1]} dimensões")

esparsidade = (1 - (matriz_tfidf.nnz / (matriz_tfidf.shape[0] * matriz_tfidf.shape[1]))) * 100
print(f"Esparsidade: {esparsidade:.2f}% dos elementos são ZERO")
print(f"Tempo de vetorização: {tempo_tfidf*1000:.2f} ms\n")

print("Ranking TF-IDF (por similaridade de cosseno):")
ranking_tfidf = sorted([(i, doc, sim) for i, (doc, sim) in enumerate(zip(documentos, similaridades_tfidf))], 
                       key=lambda x: x[2], reverse=True)
for pos, (idx, doc, sim) in enumerate(ranking_tfidf, 1):
    print(f"  {pos}. [Doc{idx+1}] Cosseno: {sim:.4f} - {doc[:60]}...")

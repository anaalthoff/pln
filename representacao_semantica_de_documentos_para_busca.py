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

# ============================================================================
# 4. ABORDAGEM SEMÂNTICA: EMBEDDINGS (REPRESENTAÇÃO DENSA)
# ============================================================================

print("\n" + "=" * 50)
print("4. ABORDAGEM SEMÂNTICA - Sentence Transformers (Vetor Denso)")
print("=" * 50)

print("Carregando modelo Sentence-BERT multilíngue...")
print("(Modelo: paraphrase-multilingual-MiniLM-L12-v2)")

modelo = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

start_time = time.time()
embeddings_docs = modelo.encode(documentos, normalize_embeddings=True)
embedding_consulta = modelo.encode([consulta], normalize_embeddings=True)
tempo_embedding = time.time() - start_time

similaridades_semanticas = cosine_similarity(embedding_consulta, embeddings_docs).flatten()

print(f"Dimensionalidade do embedding: {embeddings_docs.shape[1]} dimensões")
print(f"Tipo do vetor: DENSO (todos os {embeddings_docs.shape[1]} elementos são não-zero)")
print(f"Tempo de geração dos embeddings: {tempo_embedding*1000:.2f} ms\n")

print("Ranking Semântico (por similaridade de cosseno):")
ranking_semantico = sorted([(i, doc, sim) for i, (doc, sim) in enumerate(zip(documentos, similaridades_semanticas))], 
                           key=lambda x: x[2], reverse=True)
for pos, (idx, doc, sim) in enumerate(ranking_semantico, 1):
    print(f"  {pos}. [Doc{idx+1}] Cosseno: {sim:.4f} - {doc[:60]}...")

# ============================================================================
# 5. BUSCA HÍBRIDA (RECIPROCAL RANK FUSION - RRF)
# ============================================================================

print("\n" + "=" * 50)
print("5. BUSCA HÍBRIDA - Reciprocal Rank Fusion (RRF)")
print("=" * 50)

def reciprocal_rank_fusion(rankings, k=60):
    """
    Combina múltiplos rankings usando Reciprocal Rank Fusion (RRF).
    
    Args:
        rankings: Lista de listas, onde cada lista contém os índices dos documentos
                  em ordem de classificação (do mais relevante para o menos).
        k: Constante para suavização (tipicamente 60).
    
    Returns:
        Lista de tuplas (índice_do_documento, pontuação_rrf) ordenada por pontuação decrescente.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            if doc_idx not in scores:
                scores[doc_idx] = 0.0
            scores[doc_idx] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Obtém os rankings individuais (listas de índices, do mais relevante para o menos)
ranking_tfidf_indices = [idx for idx, _, _ in ranking_tfidf]
ranking_semantico_indices = [idx for idx, _, _ in ranking_semantico]

print("Ranking TF-IDF (índices):", ranking_tfidf_indices)
print("Ranking Semântico (índices):", ranking_semantico_indices)

# Aplica RRF
ranking_hibrido = reciprocal_rank_fusion([ranking_tfidf_indices, ranking_semantico_indices], k=60)

print("\nRanking Híbrido (RRF - combina TF-IDF + Embeddings):")
for pos, (doc_idx, score) in enumerate(ranking_hibrido, 1):
    print(f"  {pos}. [Doc{doc_idx+1}] Pontuação RRF: {score:.6f} - {documentos[doc_idx][:60]}...")

# ============================================================================
# 6. COMPARAÇÃO DETALHADA ENTRE OS RANKINGS
# ============================================================================

print("\n" + "=" * 50)
print("6. COMPARAÇÃO DETALHADA DOS RANKINGS")
print("=" * 50)

print("\n┌─────────┬──────────────┬──────────────────┬─────────────────┐")
print("│ Documento │    TF-IDF    │   Embeddings     │    Híbrido      │")
print("│           │ (posição/score)│ (posição/score)  │ (posição/score) │")
print("├───────────┼──────────────┼──────────────────┼─────────────────┤")

for doc_idx in range(len(documentos)):
    # Encontra posição no ranking TF-IDF
    tfidf_pos = next(pos for pos, (idx, _, _) in enumerate(ranking_tfidf, 1) if idx == doc_idx)
    tfidf_score = similaridades_tfidf[doc_idx]
    
    # Encontra posição no ranking semântico
    sem_pos = next(pos for pos, (idx, _, _) in enumerate(ranking_semantico, 1) if idx == doc_idx)
    sem_score = similaridades_semanticas[doc_idx]
    
    # Encontra posição no ranking híbrido
    hibrido_pos = next(pos for pos, (idx, _) in enumerate(ranking_hibrido, 1) if idx == doc_idx)
    hibrido_score = next(score for idx, score in ranking_hibrido if idx == doc_idx)
    
    print(f"│ Doc{doc_idx+1:<6} │ {tfidf_pos} ({tfidf_score:.3f})   │ {sem_pos} ({sem_score:.3f})     │ {hibrido_pos} ({hibrido_score:.4f})    │")

print("└─────────┴──────────────┴──────────────────┴─────────────────┘")

# ============================================================================
# 7. VISUALIZAÇÃO 2D DO ESPAÇO SEMÂNTICO (PCA)
# ============================================================================

print("\n" + "=" * 50)
print("7. VISUALIZAÇÃO 2D - Espaço Semântico dos Embeddings")
print("=" * 50)

# Reduz dimensionalidade para 2D com PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_docs)
consulta_2d = pca.transform(embedding_consulta)

# Cria o gráfico
fig, ax = plt.subplots(figsize=(12, 8))

# Plota documentos
for i, (x, y) in enumerate(embeddings_2d):
    # Cor baseada na similaridade semântica com a consulta
    if similaridades_semanticas[i] > 0.6:
        cor = '#2ecc71'  # verde escuro - muito relevante
        tamanho = 200
    elif similaridades_semanticas[i] > 0.3:
        cor = '#f39c12'  # laranja - moderadamente relevante
        tamanho = 160
    else:
        cor = '#e74c3c'  # vermelho - irrelevante
        tamanho = 120
    
    ax.scatter(x, y, c=cor, s=tamanho, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.annotate(f'D{i+1}\n{similaridades_semanticas[i]:.2f}', 
                (x, y), xytext=(8, 8), textcoords='offset points', 
                fontsize=9, fontweight='bold', backgroundcolor='white', 
                alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Plota consulta
ax.scatter(consulta_2d[0, 0], consulta_2d[0, 1], c='#3498db', s=400, marker='*', 
           label=f'Consulta: "{consulta}"', edgecolors='black', 
           linewidth=2, zorder=5)

ax.set_title('Espaço Semântico dos Documentos\nRedução PCA (384D → 2D) - Valores indicam similaridade de cosseno', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Componente Principal 1', fontsize=11)
ax.set_ylabel('Componente Principal 2', fontsize=11)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.show()
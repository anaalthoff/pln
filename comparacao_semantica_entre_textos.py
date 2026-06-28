# Vamos demonstrar a comparação semântica entre frases usando sentence-transformers (SBERT) e um cross-encoder para reranking. Não usaremos gensim.
# Cenário: Comparar similaridade entre frases em português, demonstrando que embeddings capturam equivalência semântica apesar de palavras diferentes, e comparar resultados com TF-IDF.

# Bibliotecas necessárias:
# sentence-transformers: Para SBERT (modelo multilíngue)
# transformers (Hugging Face): Para cross-encoder
# scikit-learn: Para similaridade do cosseno, TF-IDF e visualização (t-SNE)
# matplotlib: Para gráficos
# numpy: Para operações numéricas

# ============================================================
# IMPORTAÇÕES
# ============================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')

# Verifica se GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo usado: {device.upper()}")
print("=" * 70)

# ============================================================
# PARTE 1: SENTENCE-TRANSFORMERS (SBERT)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 1: Sentence-Transformers (SBERT - Bi-encoder)")
print("=" * 70)

# Carrega modelo multilíngue para embeddings de frases
print("\nCarregando modelo SBERT...")
modelo_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
modelo_sbert.to(device)
print("✓ Modelo carregado!")

# Conjunto de frases em português para demonstrar similaridade semântica
frases = [
    # Grupo 1: Medicina/Cirurgia (paráfrases)
    "O médico operou o paciente.",
    "O cirurgião realizou a cirurgia.",
    "O doutor fez o procedimento cirúrgico.",
    
    # Grupo 2: Direito (tópico diferente)
    "O advogado leu o contrato.",
    
    # Grupo 3: Animais (tópico diferente)
    "O gato dormiu no sofá.",
    "O cachorro correu no parque.",
    
    # Grupo 4: Variação do grupo 1
    "O paciente foi operado pelo médico.",
    
    # Grupo 5: Negação (para demonstrar contradição)
    "O médico não operou o paciente.",
    
    # Grupo 6: Relacionado mas não igual
    "A cirurgia foi um sucesso."
]

print("\nFrases analisadas:")
for i, f in enumerate(frases):
    print(f"{i+1:2d}. {f}")

# Gera embeddings das frases (SBERT)
print("\nGerando embeddings SBERT...")
embeddings_sbert = modelo_sbert.encode(frases, convert_to_numpy=True)
print(f"✓ Embeddings gerados! Shape: {embeddings_sbert.shape}")
print(f"  → {embeddings_sbert.shape[0]} frases × {embeddings_sbert.shape[1]} dimensões")

# Calcula matriz de similaridade (cosseno)
similaridades_sbert = cosine_similarity(embeddings_sbert)

print("\n" + "-" * 70)
print("MATRIZ DE SIMILARIDADE (SBERT)")
print("-" * 70)
print("     ", end="")
for i in range(1, len(frases)+1):
    print(f"  F{i:2d} ", end="")
print()
for i in range(len(frases)):
    print(f"F{i+1:2d}: ", end="")
    for j in range(len(frases)):
        print(f"{similaridades_sbert[i][j]:.3f} ", end="")
        if j == len(frases)-1:
            print()

print("\n" + "-" * 70)
print("ANÁLISE DE SIMILARIDADE SEMÂNTICA (SBERT)")
print("-" * 70)

# Análise dos resultados
sim_f1_f2 = similaridades_sbert[0][1]
sim_f1_f3 = similaridades_sbert[0][2]
sim_f1_f4 = similaridades_sbert[0][3]  # advogado
sim_f1_f5 = similaridades_sbert[0][4]  # gato
sim_f1_f7 = similaridades_sbert[0][6]  # paciente foi operado
sim_f1_f8 = similaridades_sbert[0][7]  # médico NÃO operou
sim_f1_f9 = similaridades_sbert[0][8]  # cirurgia foi sucesso

print(f"\n1. Similaridade (F1, F2): {sim_f1_f2:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[1]}'")
print(f"   → Análise: Similaridade MUITO ALTA (paráfrase perfeita)")
print(f"   → 'médico' ↔ 'cirurgião', 'operou' ↔ 'realizou cirurgia'")

print(f"\n2. Similaridade (F1, F3): {sim_f1_f3:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[2]}'")
print(f"   → Análise: Similaridade ALTA (paráfrase com reformulação)")

print(f"\n3. Similaridade (F1, F4): {sim_f1_f4:.4f}")
print(f"   → '{frases[0]}' (medicina)")
print(f"   → '{frases[3]}' (direito)")
print(f"   → Análise: Similaridade BAIXA (tópicos diferentes)")

print(f"\n4. Similaridade (F1, F5): {sim_f1_f5:.4f}")
print(f"   → '{frases[0]}' (medicina)")
print(f"   → '{frases[4]}' (gato)")
print(f"   → Análise: Similaridade MUITO BAIXA (tópicos completamente diferentes)")

print(f"\n5. Similaridade (F1, F7): {sim_f1_f7:.4f}")
print(f"   → '{frases[0]}' (médico operou paciente)")
print(f"   → '{frases[6]}' (paciente foi operado pelo médico)")
print(f"   → Análise: Similaridade MUITO ALTA (mesmo evento, voz passiva)")

print(f"\n6. Similaridade (F1, F8): {sim_f1_f8:.4f}")
print(f"   → '{frases[0]}' (médico operou paciente)")
print(f"   → '{frases[7]}' (médico NÃO operou paciente)")
print(f"   → Análise: Similaridade MODERADA (negação reduz similaridade)")

print(f"\n7. Similaridade (F1, F9): {sim_f1_f9:.4f}")
print(f"   → '{frases[0]}' (operação)")
print(f"   → '{frases[8]}' (cirurgia foi sucesso)")
print(f"   → Análise: Similaridade MÉDIA-ALTA (relacionado, mas não igual)")

# ============================================================
# PARTE 2: COMPARAÇÃO COM TF-IDF (LINHA DE BASE)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2: Comparação com TF-IDF (Linha de Base Léxica)")
print("=" * 70)

# Cria vetorizador TF-IDF
vectorizer_tfidf = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')

# Calcula matriz TF-IDF
tfidf_matrix = vectorizer_tfidf.fit_transform(frases)

# Calcula similaridade do cosseno TF-IDF
similaridades_tfidf = cosine_similarity(tfidf_matrix)

print("\nMatriz de Similaridade (TF-IDF) - valores selecionados:")
print(f"Similaridade (F1, F2) TF-IDF: {similaridades_tfidf[0][1]:.4f}")
print(f"Similaridade (F1, F3) TF-IDF: {similaridades_tfidf[0][2]:.4f}")
print(f"Similaridade (F1, F7) TF-IDF: {similaridades_tfidf[0][6]:.4f}")

print("\n" + "-" * 70)
print("COMPARAÇÃO SBERT vs. TF-IDF")
print("-" * 70)

print("\n┌───────────────────────────────┬─────────────┬─────────────┐")
print("│ Par de Frases                 │   SBERT     │   TF-IDF    │")
print("├───────────────────────────────┼─────────────┼─────────────┤")
print(f"│ F1 × F2 (paráfrase)           │ {sim_f1_f2:.4f}      │ {similaridades_tfidf[0][1]:.4f}      │")
print(f"│ F1 × F3 (paráfrase)           │ {sim_f1_f3:.4f}      │ {similaridades_tfidf[0][2]:.4f}      │")
print(f"│ F1 × F7 (voz passiva)         │ {sim_f1_f7:.4f}      │ {similaridades_tfidf[0][6]:.4f}      │")
print("└───────────────────────────────┴─────────────┴─────────────┘")

print("\n✓ CONCLUSÃO: SBERT captura similaridade SEMÂNTICA onde TF-IDF falha")
print("  (sobreposição lexical zero ou baixa)")
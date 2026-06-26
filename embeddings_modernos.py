# Bibliotecas necessárias:
# sentence-transformers: Para embeddings de frases (modelo multilíngue)
# transformers (Hugging Face): Para carregar BERTimbau
# scikit-learn: Para similaridade do cosseno e visualização (t-SNE)
# matplotlib: Para gráficos
# torch: Para processamento com BERTimbau
# numpy: Para operações numéricas

# ============================================================
# IMPORTAÇÕES
# ============================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

# Verifica se GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo usado: {device.upper()}")
print("=" * 70)

# ============================================================
# PARTE 1: EMBEDDINGS DE FRASES COM SENTENCE-TRANSFORMERS
# ============================================================
print("\n" + "=" * 70)
print("PARTE 1: Sentence-Transformers (Modelo Multilíngue)")
print("=" * 70)

# Carrega modelo multilíngue otimizado para similaridade semântica
# Este modelo suporta português e várias outras línguas
print("\nCarregando modelo Sentence-Transformer...")
modelo_frases = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
modelo_frases.to(device)
print("✓ Modelo carregado com sucesso!")

# Frases de exemplo em português para demonstrar similaridade semântica
frases = [
    "O médico operou o paciente.",           # Frase 1 - médio/cirurgia
    "O cirurgião realizou a cirurgia.",      # Frase 2 - mesmo significado da F1
    "O advogado leu o contrato.",            # Frase 3 - domínio diferente
    "O gato dormiu no sofá.",                # Frase 4 - animal doméstico
    "O cachorro correu no parque.",          # Frase 5 - animal doméstico
    "O médico examinou o paciente cuidadosamente."  # Frase 6 - similar à F1
]

print("\nFrases analisadas:")
for i, f in enumerate(frases, 1):
    print(f"  F{i}: {f}")

# Gera embeddings das frases (vetores densos de 384 dimensões)
print("\nGerando embeddings das frases...")
embeddings = modelo_frases.encode(frases, convert_to_numpy=True)
print(f"✓ Embeddings gerados! Shape: {embeddings.shape}")
print(f"  → {embeddings.shape[0]} frases × {embeddings.shape[1]} dimensões")

# Calcula matriz de similaridade (cosseno) entre todas as frases
similaridades = cosine_similarity(embeddings)

print("\n" + "-" * 70)
print("Matriz de Similaridade do Cosseno (valores arredondados):")
print("-" * 70)
print("     ", end="")
for i in range(1, 7):
    print(f"  F{i}   ", end="")
print()
for i in range(6):
    print(f"F{i+1}: ", end="")
    for j in range(6):
        print(f"{similaridades[i][j]:.3f} ", end="")
        if j == 5:
            print()

print("\n" + "-" * 70)
print("ANÁLISE DE SIMILARIDADE SEMÂNTICA")
print("-" * 70)

# Análise dos resultados
sim_f1_f2 = similaridades[0][1]
sim_f1_f6 = similaridades[0][5]
sim_f1_f3 = similaridades[0][2]
sim_f4_f5 = similaridades[3][4]
sim_f2_f3 = similaridades[1][2]

print(f"\n1. Similaridade (F1, F2): {sim_f1_f2:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[1]}'")
print(f"   → Análise: Similaridade ALTA (valores próximos de 0.8-0.9)")
print(f"   → As frases têm MESMO SIGNIFICADO, mesmo com palavras diferentes!")
print(f"   → Exemplo: 'médico' ≈ 'cirurgião', 'operou' ≈ 'realizou cirurgia'")

print(f"\n2. Similaridade (F1, F6): {sim_f1_f6:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[5]}'")
print(f"   → Análise: Similaridade MUITO ALTA")
print(f"   → Ambas sobre médico/paciente, com variação lexical")

print(f"\n3. Similaridade (F1, F3): {sim_f1_f3:.4f}")
print(f"   → '{frases[0]}' (medicina)")
print(f"   → '{frases[2]}' (direito)")
print(f"   → Análise: Similaridade BAIXA (domínios diferentes)")

print(f"\n4. Similaridade (F4, F5): {sim_f4_f5:.4f}")
print(f"   → '{frases[3]}' (gato)")
print(f"   → '{frases[4]}' (cachorro)")
print(f"   → Análise: Similaridade MÉDIA-ALTA")
print(f"   → Ambos são animais domésticos, semanticamente próximos")

print(f"\n5. Similaridade (F2, F3): {sim_f2_f3:.4f}")
print(f"   → '{frases[1]}' (cirurgia)")
print(f"   → '{frases[2]}' (direito)")
print(f"   → Análise: Similaridade BAIXA (domínios diferentes)")

# Bibliotecas necessárias:
# spacy: Para carregar embeddings de palavras (modelo pt_core_news_lg).
# sentence-transformers: Para embeddings de frases (modelo paraphrase-multilingual-MiniLM-L12-v2).
# scikit-learn: Para calcular a similaridade do cosseno.
# matplotlib e sklearn.manifold: Para visualização 2D (t-SNE).

# Importações
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer

# 1. CARREGAR MODELOS
print("Carregando modelos...")
# Modelo de palavras do spaCy (Português)
# Certifique-se de ter baixado: python -m spacy download pt_core_news_lg
nlp = spacy.load("pt_core_news_lg")

# Modelo de frases do SentenceTransformer (Multilíngue)
# Ideal para comparar frases inteiras
modelo_frases = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Modelos carregados!")

# 2. EXEMPLO 1: SIMILARIDADE ENTRE PALAVRAS (SPACY)
print("\n--- Similaridade entre Palavras (spaCy) ---")
palavras = ["carro", "automóvel", "gato", "cachorro", "computador"]
# Obtém os vetores (embeddings) das palavras
vetores_palavras = {p: nlp(p).vector for p in palavras}

# Calcula similaridade (cosseno) entre 'carro' e outras
carro_vec = vetores_palavras["carro"].reshape(1, -1) # reshape para 2D
automovel_vec = vetores_palavras["automóvel"].reshape(1, -1)
gato_vec = vetores_palavras["gato"].reshape(1, -1)

sim_carro_automovel = cosine_similarity(carro_vec, automovel_vec)[0][0]
sim_carro_gato = cosine_similarity(carro_vec, gato_vec)[0][0]

print(f"Similaridade ('carro', 'automóvel'): {sim_carro_automovel:.4f}")
print(f"Similaridade ('carro', 'gato'): {sim_carro_gato:.4f}")
print("Observação: 'carro' e 'automóvel' são muito mais similares, como esperado pela Hipótese Distribucional!")

# 3. EXEMPLO 2: SIMILARIDADE ENTRE FRASES (SENTENCE TRANSFORMERS)
print("\n--- Similaridade entre Frases (SentenceTransformer) ---")
frases = [
    "O médico operou o paciente.",
    "O cirurgião realizou a cirurgia.",
    "O advogado leu o contrato."
]

# Gera os embeddings das frases (vetores densos)
embeddings_frases = modelo_frases.encode(frases)

# Calcula a matriz de similaridade entre todas as frases
similaridades = cosine_similarity(embeddings_frases)

print("Matriz de Similaridade (frase 0, 1, 2):")
print(similaridades)
print(f"\nSimilaridade (Frase 0, Frase 1): {similaridades[0][1]:.4f}")
print(f"Similaridade (Frase 0, Frase 2): {similaridades[0][2]:.4f}")
print("A maior similaridade entre 'médico/operou/paciente' e 'cirurgião/realizou/cirurgia' demonstra a captura semântica, apesar das palavras serem diferentes!")

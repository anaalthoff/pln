# Importações
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Carregar modelo spaCy para português e obter stopwords
nlp = spacy.load('pt_core_news_sm')
stopwords_pt = list(nlp.Defaults.stop_words)

# 1. Corpus de documentos em português
documentos = [
    "Inteligência artificial é usada em carros autônomos",
    "Machine learning permite que computadores aprendam com dados",
    "Processamento de linguagem natural ajuda chatbots a entender humanos",
    "Carros elétricos e autônomos são o futuro do transporte",
    "O tempo está ensolarado hoje, ótimo para passear"
]

# 2. Criação do modelo TF-IDF com stopwords do spaCy
vectorizer = TfidfVectorizer(stop_words=stopwords_pt, lowercase=True)

# 3. Aplica o modelo ao corpus e gera a matriz TF-IDF
matriz_tfidf = vectorizer.fit_transform(documentos)

# 4. Exibe informações sobre a matriz
print("=" * 60)
print("INFORMAÇÕES SOBRE A REPRESENTAÇÃO NUMÉRICA")
print("=" * 60)
print(f"Número de documentos no corpus: {matriz_tfidf.shape[0]}")
print(f"Tamanho do vocabulário (número de termos únicos): {matriz_tfidf.shape[1]}")
print(f"Exemplo de termos encontrados: {vectorizer.get_feature_names_out()[:10]}")
print(f"\nMatriz TF-IDF (amostra densa, mas ela é esparsa):")
print(matriz_tfidf.toarray())
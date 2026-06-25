# Download dos recursos do NLTK e spaCy
import nltk
nltk.download('punkt')
nltk.download('rslp') # Stemmer para português

# Importações
import nltk
import spacy
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# Carregar modelos
stemmer = RSLPStemmer()
nlp = spacy.load('pt_core_news_sm')

palavras = ["correndo", "correram", "melhores", "péssima", "inconsistência"]

print("Palavra Original | Stemming (RSLP) | Lematização (spaCy)")
print("-" * 60)
for palavra in palavras:
    stem = stemmer.stem(palavra)
    doc = nlp(palavra)
    lema = doc[0].lemma_  # Pega o lema do primeiro token
    print(f"{palavra:15} | {stem:15} | {lema}")

# Como o pré-processamento (incluindo stemming e lematização) afeta a similaridade entre duas frases semanticamente próximas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Frases de exemplo
frase1 = "O médico operou o paciente com sucesso."
frase2 = "Os médicos operaram os pacientes com sucesso."
frase3 = "O computador processa dados rapidamente." # Frase não relacionada

# Função para aplicar stemming em uma frase
def aplicar_stemming(frase):
    tokens = word_tokenize(frase, language='portuguese')
    tokens_stem = [stemmer.stem(token) for token in tokens if token.isalnum()] # remove pontuação
    return ' '.join(tokens_stem)

# Função para aplicar lematização em uma frase
def aplicar_lematizacao(frase):
    doc = nlp(frase)
    tokens_lema = [token.lemma_ for token in doc if not token.is_punct and token.lemma_.strip()]
    return ' '.join(tokens_lema)

# Preparar as versões das frases
versoes = {
    "Original": [frase1, frase2, frase3],
    "Stemming": [aplicar_stemming(frase1), aplicar_stemming(frase2), aplicar_stemming(frase3)],
    "Lematização": [aplicar_lematizacao(frase1), aplicar_lematizacao(frase2), aplicar_lematizacao(frase3)]
}

# Calcular similaridade TF-IDF
resultados = {}
for nome, textos in versoes.items():
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    # Similaridade entre frase1 (índice 0) e frase2 (índice 1)
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    resultados[nome] = sim

# Exibir resultados
print("Similaridade (Frase1 x Frase2) por tipo de normalização:")
for nome, sim in resultados.items():
    print(f"{nome}: {sim:.4f}")

import matplotlib.pyplot as plt

nomes = list(resultados.keys())
similaridades = list(resultados.values())

plt.figure(figsize=(8, 5))
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.bar(nomes, similaridades, color=cores)
plt.ylabel('Similaridade de Cosseno (TF-IDF)')
plt.title('Impacto do Stemming e Lematização na Similaridade entre Frases')
plt.ylim(0, 1)
for i, v in enumerate(similaridades):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()
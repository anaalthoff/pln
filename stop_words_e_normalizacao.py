# Download dos recursos do NLTK e spaCy
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Importações
import re
import nltk
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar modelos
nlp = spacy.load('pt_core_news_sm')
modelo_embedding = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # Modelo multilíngue
stopwords_nltk = set(stopwords.words('portuguese'))

# Pré-processamento
def preprocessar_texto(texto, remover_stopwords=True, lematizar=True):
    """
    Aplica normalização e opcionalmente remove stopwords e lematiza.
    """
    # 1. Normalização inicial (case folding e remoção de caracteres especiais/pontuação)
    texto = texto.lower()
    # Remove pontuação, números e caracteres especiais, mantém letras e espaços
    texto = re.sub(r'[^a-záéíóúâêîôûãõç\s]', '', texto)

    # 2. Tokenização e Lematização com spaCy (ou NLTK)
    doc = nlp(texto)
    tokens_processados = []

    for token in doc:
        # Decide se usa o lema ou o texto original
        if lematizar:
            palavra_base = token.lemma_
        else:
            palavra_base = token.text

        # 3. Remoção de stopwords (se aplicável)
        if remover_stopwords:
            if palavra_base not in stopwords_nltk and len(palavra_base) > 2:
                tokens_processados.append(palavra_base)
        else:
            if len(palavra_base) > 2: # remove tokens muito curtos (ex: 'a', 'e')
                tokens_processados.append(palavra_base)

    return " ".join(tokens_processados)

# Exemplo de uso
frase_original = "Os alunos do curso de Ciência da Computação estudam técnicas avançadas de PLN!"
frase_sem_stopwords = preprocessar_texto(frase_original, remover_stopwords=True, lematizar=True)
frase_sem_preprocess = preprocessar_texto(frase_original, remover_stopwords=False, lematizar=False)

print("Original:", frase_original)
print("Sem stopwords e lematizado:", frase_sem_stopwords)
print("Apenas normalizado (sem remoção de stopwords):", frase_sem_preprocess)
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
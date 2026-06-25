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
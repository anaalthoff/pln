import nltk
import spacy
import re

# Baixe os dados do NLTK para tokenização (execute uma vez)
# nltk.download('punkt')
# nltk.download('punkt_tab') # Para versões mais novas do Nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Carregue o modelo do spaCy para português
nlp = spacy.load("pt_core_news_sm")

frase_exemplo = "Dr. Paulo, o aluno da Ufba, comprou um notebook de R$ 2.500,00? Incrível!"

print(f"Frase original: {frase_exemplo}\n")

# Tokenização ingênua por espaços
tokens_espacos = frase_exemplo.split()
print("1. Tokenização por espaços (split()):")
print(tokens_espacos)

# Saída: ['Dr.', 'Paulo,', 'o', 'aluno', 'da', 'Ufba,', 'comprou', 'um', 'notebook', 'de', 'R$', '2.500,00?', 'Incrível!']
# Problemas: Pontuação (',', '?', '!') grudada nas palavras.
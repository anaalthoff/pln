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

# O word_tokenize do NLTK já separa pontuação, mas não lida bem com contrações e abreviações em português por padrão (o modelo punkt é treinado majoritariamente para inglês). Ele trata "Dr." como uma palavra e "da" como um token único.

nltk.download('punkt_tab')
# Tokenização NLTK (já separa pontuação)
tokens_nltk = word_tokenize(frase_exemplo, language='portuguese')
print("\n2. Tokenização com NLTK (word_tokenize):")
print(tokens_nltk)

# Saída esperada: ['Dr.', 'Paulo', ',', 'o', 'aluno', 'da', 'Ufba', ',', 'comprou', 'um', 'notebook', 'de', 'R$', '2.500,00', '?', 'Incrível', '!'

# Tokenização Avançada com spaCy
doc = nlp(frase_exemplo)
tokens_spacy = [token.text for token in doc]
print("\n3. Tokenização com spaCy:")
print(tokens_spacy)
# Saída esperada: ['Dr.', 'Paulo', ',', 'o', 'aluno', 'de', 'a', 'Ufba', ',', 'comprou', 'um', 'notebook', 'de', 'R$', '2.500,00', '?', 'Incrível', '!']

# Análise detalhada de um token
print("\n--- Análise Detalhada do Token 'da' (spaCy) ---")
for token in doc:
    if token.text == 'da':
        print(f"Texto: '{token.text}', Lema: '{token.lemma_}', POS: '{token.pos_}'")
        # Saída: Texto: 'da', Lema: 'de', POS: 'ADP'
        # O spaCy já expandiu a contração 'da' em 'de' + 'a', mas mantém o texto original 'da'.
        # A propriedade .lemma_ retorna a forma base ('de').
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

# Visualização da Frequência de Tokens
import matplotlib.pyplot as plt
from collections import Counter

# Pequeno corpus de exemplo
corpus = [
    "O rato roeu a roupa do rei de Roma.",
    "A rainha roeu a roupa do rato.",
    "O rei ficou muito irritado com o rato!"
]

# Tokenização simples com NLTK
todos_tokens = []
for sentenca in corpus:
    # Tokeniza e converte para minúsculas para melhor contagem
    tokens = word_tokenize(sentenca.lower(), language='portuguese')
    todos_tokens.extend(tokens)

# Contagem de frequência
freq = Counter(todos_tokens)

# Separando tokens que são palavras vs. pontuação
palavras = {token: count for token, count in freq.items() if token.isalpha()}
pontuacao = {token: count for token, count in freq.items() if not token.isalpha()}

# Visualização
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de palavras mais comuns
palavras_comuns = Counter(palavras).most_common(5)
ax1.bar([p[0] for p in palavras_comuns], [p[1] for p in palavras_comuns], color='skyblue')
ax1.set_title('Top 5 Palavras (Tokens Alfabéticos)')
ax1.set_xlabel('Palavra')
ax1.set_ylabel('Frequência')

# Gráfico de pontuação
ax2.bar(pontuacao.keys(), pontuacao.values(), color='lightcoral')
ax2.set_title('Frequência de Pontuação (Tokens Não-Alfabéticos)')
ax2.set_xlabel('Pontuação')
ax2.set_ylabel('Frequência')

plt.tight_layout()
plt.show()

print("Frequência de Pontuação:", pontuacao)
# Saída esperada: Frequência de Pontuação: {'.': 3, ',': 0, '!': 1}
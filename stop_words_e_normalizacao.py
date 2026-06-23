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

# Impacto na Similaridade Semântica
# Vamos comparar como diferentes níveis de pré-processamento afetam a similaridade entre duas frases semanticamente próximas, mas com estruturas diferentes.
# Frases de exemplo
frase1 = "O médico operou o paciente com sucesso."
frase2 = "A cirurgia realizada pelo doutor no enfermo foi bem-sucedida."

# Aplica diferentes níveis de pré-processamento
# Caso A: Sem pré-processamento (texto bruto)
# Caso B: Com normalização e lematização, mas com stopwords
# Caso C: Com normalização, lematização e remoção de stopwords

preprocess_casos = {
    "A - Bruto": lambda x: x,
    "B - Normalizado (c/ stopwords)": lambda x: preprocessar_texto(x, remover_stopwords=False, lematizar=True),
    "C - Normalizado (s/ stopwords)": lambda x: preprocessar_texto(x, remover_stopwords=True, lematizar=True)
}

# Calcula embeddings e similaridade de cosseno
resultados = {}
for nome, func in preprocess_casos.items():
    p1 = func(frase1)
    p2 = func(frase2)
    emb1 = modelo_embedding.encode(p1, convert_to_tensor=True)
    emb2 = modelo_embedding.encode(p2, convert_to_tensor=True)
    similaridade = util.pytorch_cos_sim(emb1, emb2).item()
    resultados[nome] = {"sim": similaridade, "p1": p1, "p2": p2}

# Exibe resultados
for nome, dados in resultados.items():
    print(f"\n--- {nome} ---")
    print(f"Frase1: {dados['p1']}")
    print(f"Frase2: {dados['p2']}")
    print(f"Similaridade de Cosseno: {dados['sim']:.4f}")

# Cria um gráfico de barras
labels = list(resultados.keys())
similaridades = [dados["sim"] for dados in resultados.values()]

plt.figure(figsize=(8, 5))
plt.bar(labels, similaridades, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Similaridade de Cosseno')
plt.title('Impacto do Pré-processamento na Similaridade Semântica')
plt.ylim(0, 1)
for i, v in enumerate(similaridades):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()
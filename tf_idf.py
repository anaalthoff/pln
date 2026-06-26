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

# 5. Calcula a matriz de similaridade por cosseno entre todos os documentos
similaridade = cosine_similarity(matriz_tfidf)

print("\n" + "=" * 60)
print("MATRIZ DE SIMILARIDADE POR COSSENO (Documento x Documento)")
print("=" * 60)
print("Linha i, Coluna j = Similaridade entre Documento i e Documento j")
print(np.round(similaridade, 3))

# 6. Visualização dos resultados em 2D com PCA
pca = PCA(n_components=2)
matriz_2d = pca.fit_transform(matriz_tfidf.toarray())

plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, doc in enumerate(documentos):
    nome_doc = f"Doc {i+1}"
    plt.scatter(matriz_2d[i, 0], matriz_2d[i, 1], c=colors[i], s=100, label=nome_doc)
    plt.annotate(nome_doc, (matriz_2d[i, 0] + 0.02, matriz_2d[i, 1] + 0.02), fontsize=12)

plt.title("Visualização 2D de Documentos no Espaço TF-IDF (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)
plt.show()

# 7. Função de busca simples
def buscar_documento(consulta, top_n=2):
    vetor_consulta = vectorizer.transform([consulta])
    similaridades = cosine_similarity(vetor_consulta, matriz_tfidf).flatten()
    indices_top = np.argsort(similaridades)[::-1][:top_n]
    
    print("\n" + "=" * 60)
    print(f"RESULTADOS DA BUSCA POR: '{consulta}'")
    print("=" * 60)
    for idx in indices_top:
        print(f"Documento {idx+1}: '{documentos[idx]}'")
        print(f"Similaridade: {similaridades[idx]:.4f}\n")

# Exemplo de busca
buscar_documento("aprendizado de máquina com computadores")
buscar_documento("transporte autônomo elétrico")
buscar_documento("clima ensolarado")
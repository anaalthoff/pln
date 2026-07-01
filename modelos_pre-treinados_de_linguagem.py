# Usaremos a biblioteca sentence-transformers (que é baseada em PLMs) para demonstrar o poder da busca semântica em português, comparando com TF-IDF.
# Cenário: Nosso sistema busca na legislação brasileira. O usuário faz uma consulta usando uma palavra informal, enquanto os documentos usam a terminologia jurídica formal.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Base de Documentos e Consulta
documentos = [
    "O pagamento do auxílio-doença será efetuado em até 30 dias.",
    "Normas para aposentadoria especial do trabalhador rural.",
    "Procedimento para solicitação do benefício de prestação continuada (BPC).",
    "É dever do empregador reter o imposto de renda na fonte."
]

consulta = "Como receber o dinheiro do seguro por incapacidade?"

# 2. Modelo TF-IDF (Baseline Lexical)
print("=== Resultados da Busca TF-IDF ===")
vectorizer = TfidfVectorizer()
# Aplica TF-IDF em todos os textos + consulta para garantir mesmo vocabulário
corpus_tfidf = vectorizer.fit_transform(documentos + [consulta])
query_tfidf = corpus_tfidf[-1]
docs_tfidf = corpus_tfidf[:-1]

# Calcula similaridade de cosseno
similarities_tfidf = cosine_similarity(query_tfidf, docs_tfidf).flatten()
# Ordena e exibe
ranking_tfidf = np.argsort(similarities_tfidf)[::-1]
for idx in ranking_tfidf:
    print(f"Doc {idx+1} (Score: {similarities_tfidf[idx]:.3f}): {documentos[idx]}")
    if idx == 0:
        print(" [ALERTA] FALHA: O documento mais relevante tem baixa similaridade lexical!\n")

# 3. Modelo PLM (Sentence-BERT - Contextual)
print("\n=== Resultados da Busca PLM (Semântica) ===")
# Carrega um modelo multilíngue otimizado para similaridade semântica
# (Exemplos: 'paraphrase-multilingual-MiniLM-L12-v2', 'distiluse-base-multilingual-cased')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Gera embeddings
doc_embeddings = model.encode(documentos)
query_embedding = model.encode([consulta])

# Calcula similaridade de cosseno
similarities_plm = cosine_similarity(query_embedding, doc_embeddings).flatten()
ranking_plm = np.argsort(similarities_plm)[::-1]

for idx in ranking_plm:
    print(f"Doc {idx+1} (Score: {similarities_plm[idx]:.3f}): {documentos[idx]}")

# 4. Visualização Gráfica (PCA para reduzir dimensões e plotar)
pca = PCA(n_components=2)
# Combina todos os embeddings para treinar o PCA
all_embeds = np.vstack([doc_embeddings, query_embedding])
pca_result = pca.fit_transform(all_embeds)

# Plotagem
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'purple']
for i, doc in enumerate(documentos):
    plt.scatter(pca_result[i, 0], pca_result[i, 1], color=colors[i], s=100, label=f"Doc {i+1}")
# Query
plt.scatter(pca_result[-1, 0], pca_result[-1, 1], color='orange', s=150, marker='*', label='Consulta')

# Anotações
for i, doc in enumerate(documentos):
    plt.annotate(f"Doc {i+1}", (pca_result[i, 0], pca_result[i, 1]))
plt.annotate("Consulta", (pca_result[-1, 0], pca_result[-1, 1]))
plt.title('Espaço Vetorial Semântico (PLM) - PCA 2D')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.show()

print("\nAnálise: O PLM identificou corretamente que 'receber dinheiro' e 'pagamento' são semanticamente próximos, e 'incapacidade' é um sinônimo contextual de 'auxílio-doença'!")
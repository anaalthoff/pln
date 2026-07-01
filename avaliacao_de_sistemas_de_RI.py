import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# 1. Carregar modelo semântico (pré-treinado)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 2. Corpus de documentos em português
documentos = [
    "Redes neurais convolucionais são ótimas para reconhecimento de imagens.",
    "Carros elétricos estão se tornando cada vez mais populares no Brasil.",
    "CNNs são arquiteturas fundamentais em visão computacional.",
    "Aprendizado profundo utiliza backpropagation para ajustar pesos.",
    "Automóveis movidos a bateria são uma alternativa sustentável.",
    "As redes neurais convolucionais (CNN) são usadas em classificação de imagens.",
    "O mercado de veículos elétricos cresceu 40% no último ano.",
    "Deep learning é uma subárea do aprendizado de máquina."
]

# 3. Ground truth: quais documentos são relevantes para a consulta "redes neurais convolucionais"?
# Índices: 0, 2, 5 são relevantes (documentos que mencionam CNNs)
relevantes_verdade = [True, False, True, False, False, True, False, False]

# 4. Função de busca semântica com limiar
def busca_semantica(consulta, documentos, modelo, limiar):
    """
    Retorna documentos cuja similaridade de cosseno com a consulta >= limiar.
    Também retorna a lista ordenada por similaridade decrescente.
    """
    embeddings_docs = modelo.encode(documentos, convert_to_tensor=True)
    emb_consulta = modelo.encode(consulta, convert_to_tensor=True)
    similaridades = util.cos_sim(emb_consulta, embeddings_docs)[0]
    
    # Ordenar documentos por similaridade decrescente
    pares = [(i, sim.item()) for i, sim in enumerate(similaridades)]
    pares_ordenados = sorted(pares, key=lambda x: x[1], reverse=True)
    
    indices_ordenados, sims_ordenadas = zip(*pares_ordenados)
    
    # Aplicar limiar
    indices_recuperados = [i for i, sim in pares_ordenados if sim >= limiar]
    
    return indices_recuperados, indices_ordenados, sims_ordenadas

# 5. Função para calcular métricas
def calcular_metricas(indices_recuperados, indices_ordenados, relevantes_verdade):
    """Calcula Precisão, Revocação, AP e Precision@K."""
    # Para precisão/revocação simples (limiar binário)
    vp = sum(1 for i in indices_recuperados if relevantes_verdade[i])
    fp = len(indices_recuperados) - vp
    fn = sum(1 for i, rel in enumerate(relevantes_verdade) if rel and i not in indices_recuperados)
    
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
    revocacao = vp / (vp + fn) if (vp + fn) > 0 else 0
    
    # Average Precision (AP) - considerando ordenação
    relevantes_encontrados = 0
    precisoes_ap = []
    for pos, idx in enumerate(indices_ordenados, start=1):
        if relevantes_verdade[idx]:
            relevantes_encontrados += 1
            precisao_pos = relevantes_encontrados / pos
            precisoes_ap.append(precisao_pos)
    total_relevantes = sum(relevantes_verdade)
    ap = np.mean(precisoes_ap) if precisoes_ap else 0
    
    # Precision@K (K=1,3,5)
    prec_k = {}
    for k in [1, 3, 5]:
        top_k = indices_ordenados[:k]
        relevantes_top_k = sum(1 for i in top_k if relevantes_verdade[i])
        prec_k[k] = relevantes_top_k / k
    
    return precisao, revocacao, ap, prec_k

# 6. Executar para diferentes limiares
consulta = "redes neurais convolucionais"
limiares = np.arange(0.1, 0.9, 0.1)
resultados = []

print("=" * 60)
print(f"Consulta: '{consulta}'")
print("=" * 60)

for limiar in limiares:
    rec, ordenados, sims = busca_semantica(consulta, documentos, model, limiar)
    p, r, ap, prec_k = calcular_metricas(rec, ordenados, relevantes_verdade)
    resultados.append((limiar, p, r, ap))
    
    print(f"\nLimiar = {limiar:.1f}")
    print(f"  Documentos recuperados: {rec}")
    print(f"  Precisão: {p:.3f}")
    print(f"  Revocação: {r:.3f}")
    print(f"  AP: {ap:.3f}")
    print(f"  Precision@1/3/5: {prec_k[1]:.1f}/{prec_k[3]:.1f}/{prec_k[5]:.1f}")

# 7. Visualização da Curva Precisão-Revocação
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for limiar, p, r, ap in resultados:
    plt.plot(r, p, 'bo', markersize=8)
plt.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Trade-off teórico')
plt.xlabel('Revocação')
plt.ylabel('Precisão')
plt.title('Curva Precisão-Revocação\n(variando o limiar de similaridade)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
for limiar, p, r, ap in resultados:
    plt.annotate(f'{limiar:.1f}', (r, p), textcoords="offset points", xytext=(5,5), ha='center')
plt.legend()

plt.subplot(1, 2, 2)
limiares_plot, ap_values = zip(*[(l, ap) for l, p, r, ap in resultados])
plt.bar(range(len(limiares_plot)), ap_values, tick_label=[f'{l:.1f}' for l in limiares_plot])
plt.xlabel('Limiar de similaridade')
plt.ylabel('Average Precision (AP)')
plt.title('AP em função do limiar')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 8. Visualização em 2D dos embeddings
from sklearn.decomposition import PCA

embeddings = model.encode(documentos, convert_to_tensor=False)
consulta_emb = model.encode(consulta, convert_to_tensor=False)

# Reduzir para 2D com PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
consulta_2d = pca.transform([consulta_emb])[0]

plt.figure(figsize=(10, 8))
# Plotar documentos
for i, (x, y) in enumerate(embeddings_2d):
    cor = 'green' if relevantes_verdade[i] else 'red'
    plt.scatter(x, y, c=cor, s=100, edgecolors='black', alpha=0.7)
    plt.annotate(f'Doc{i}', (x, y), fontsize=9)

# Plotar consulta
plt.scatter(consulta_2d[0], consulta_2d[1], c='blue', s=200, marker='*', 
            edgecolors='black', label='Consulta')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização dos Embeddings em 2D (PCA)\nVerde=Relevante, Vermelho=Não Relevante')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
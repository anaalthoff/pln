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
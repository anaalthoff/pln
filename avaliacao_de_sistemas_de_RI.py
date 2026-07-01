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
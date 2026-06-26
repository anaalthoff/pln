# Bibliotecas necessárias:
# sentence-transformers: Para embeddings de frases (modelo multilíngue)
# transformers (Hugging Face): Para carregar BERTimbau
# scikit-learn: Para similaridade do cosseno e visualização (t-SNE)
# matplotlib: Para gráficos
# torch: Para processamento com BERTimbau
# numpy: Para operações numéricas

# ============================================================
# IMPORTAÇÕES
# ============================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

# Verifica se GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo usado: {device.upper()}")
print("=" * 70)
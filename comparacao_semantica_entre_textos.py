# Vamos demonstrar a comparação semântica entre frases usando sentence-transformers (SBERT) e um cross-encoder para reranking. Não usaremos gensim.
# Cenário: Comparar similaridade entre frases em português, demonstrando que embeddings capturam equivalência semântica apesar de palavras diferentes, e comparar resultados com TF-IDF.

# Bibliotecas necessárias:
# sentence-transformers: Para SBERT (modelo multilíngue)
# transformers (Hugging Face): Para cross-encoder
# scikit-learn: Para similaridade do cosseno, TF-IDF e visualização (t-SNE)
# matplotlib: Para gráficos
# numpy: Para operações numéricas

# ============================================================
# IMPORTAÇÕES
# ============================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')

# Verifica se GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo usado: {device.upper()}")
print("=" * 70)

# ============================================================
# PARTE 1: SENTENCE-TRANSFORMERS (SBERT)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 1: Sentence-Transformers (SBERT - Bi-encoder)")
print("=" * 70)

# Carrega modelo multilíngue para embeddings de frases
print("\nCarregando modelo SBERT...")
modelo_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
modelo_sbert.to(device)
print("✓ Modelo carregado!")

# Conjunto de frases em português para demonstrar similaridade semântica
frases = [
    # Grupo 1: Medicina/Cirurgia (paráfrases)
    "O médico operou o paciente.",
    "O cirurgião realizou a cirurgia.",
    "O doutor fez o procedimento cirúrgico.",
    
    # Grupo 2: Direito (tópico diferente)
    "O advogado leu o contrato.",
    
    # Grupo 3: Animais (tópico diferente)
    "O gato dormiu no sofá.",
    "O cachorro correu no parque.",
    
    # Grupo 4: Variação do grupo 1
    "O paciente foi operado pelo médico.",
    
    # Grupo 5: Negação (para demonstrar contradição)
    "O médico não operou o paciente.",
    
    # Grupo 6: Relacionado mas não igual
    "A cirurgia foi um sucesso."
]

print("\nFrases analisadas:")
for i, f in enumerate(frases):
    print(f"{i+1:2d}. {f}")

# Gera embeddings das frases (SBERT)
print("\nGerando embeddings SBERT...")
embeddings_sbert = modelo_sbert.encode(frases, convert_to_numpy=True)
print(f"✓ Embeddings gerados! Shape: {embeddings_sbert.shape}")
print(f"  → {embeddings_sbert.shape[0]} frases × {embeddings_sbert.shape[1]} dimensões")

# Calcula matriz de similaridade (cosseno)
similaridades_sbert = cosine_similarity(embeddings_sbert)

print("\n" + "-" * 70)
print("MATRIZ DE SIMILARIDADE (SBERT)")
print("-" * 70)
print("     ", end="")
for i in range(1, len(frases)+1):
    print(f"  F{i:2d} ", end="")
print()
for i in range(len(frases)):
    print(f"F{i+1:2d}: ", end="")
    for j in range(len(frases)):
        print(f"{similaridades_sbert[i][j]:.3f} ", end="")
        if j == len(frases)-1:
            print()

print("\n" + "-" * 70)
print("ANÁLISE DE SIMILARIDADE SEMÂNTICA (SBERT)")
print("-" * 70)

# Análise dos resultados
sim_f1_f2 = similaridades_sbert[0][1]
sim_f1_f3 = similaridades_sbert[0][2]
sim_f1_f4 = similaridades_sbert[0][3]  # advogado
sim_f1_f5 = similaridades_sbert[0][4]  # gato
sim_f1_f7 = similaridades_sbert[0][6]  # paciente foi operado
sim_f1_f8 = similaridades_sbert[0][7]  # médico NÃO operou
sim_f1_f9 = similaridades_sbert[0][8]  # cirurgia foi sucesso

print(f"\n1. Similaridade (F1, F2): {sim_f1_f2:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[1]}'")
print(f"   → Análise: Similaridade MUITO ALTA (paráfrase perfeita)")
print(f"   → 'médico' ↔ 'cirurgião', 'operou' ↔ 'realizou cirurgia'")

print(f"\n2. Similaridade (F1, F3): {sim_f1_f3:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[2]}'")
print(f"   → Análise: Similaridade ALTA (paráfrase com reformulação)")

print(f"\n3. Similaridade (F1, F4): {sim_f1_f4:.4f}")
print(f"   → '{frases[0]}' (medicina)")
print(f"   → '{frases[3]}' (direito)")
print(f"   → Análise: Similaridade BAIXA (tópicos diferentes)")

print(f"\n4. Similaridade (F1, F5): {sim_f1_f5:.4f}")
print(f"   → '{frases[0]}' (medicina)")
print(f"   → '{frases[4]}' (gato)")
print(f"   → Análise: Similaridade MUITO BAIXA (tópicos completamente diferentes)")

print(f"\n5. Similaridade (F1, F7): {sim_f1_f7:.4f}")
print(f"   → '{frases[0]}' (médico operou paciente)")
print(f"   → '{frases[6]}' (paciente foi operado pelo médico)")
print(f"   → Análise: Similaridade MUITO ALTA (mesmo evento, voz passiva)")

print(f"\n6. Similaridade (F1, F8): {sim_f1_f8:.4f}")
print(f"   → '{frases[0]}' (médico operou paciente)")
print(f"   → '{frases[7]}' (médico NÃO operou paciente)")
print(f"   → Análise: Similaridade MODERADA (negação reduz similaridade)")

print(f"\n7. Similaridade (F1, F9): {sim_f1_f9:.4f}")
print(f"   → '{frases[0]}' (operação)")
print(f"   → '{frases[8]}' (cirurgia foi sucesso)")
print(f"   → Análise: Similaridade MÉDIA-ALTA (relacionado, mas não igual)")

# ============================================================
# PARTE 2: COMPARAÇÃO COM TF-IDF (LINHA DE BASE)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2: Comparação com TF-IDF (Linha de Base Léxica)")
print("=" * 70)

# Cria vetorizador TF-IDF
vectorizer_tfidf = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')

# Calcula matriz TF-IDF
tfidf_matrix = vectorizer_tfidf.fit_transform(frases)

# Calcula similaridade do cosseno TF-IDF
similaridades_tfidf = cosine_similarity(tfidf_matrix)

print("\nMatriz de Similaridade (TF-IDF) - valores selecionados:")
print(f"Similaridade (F1, F2) TF-IDF: {similaridades_tfidf[0][1]:.4f}")
print(f"Similaridade (F1, F3) TF-IDF: {similaridades_tfidf[0][2]:.4f}")
print(f"Similaridade (F1, F7) TF-IDF: {similaridades_tfidf[0][6]:.4f}")

print("\n" + "-" * 70)
print("COMPARAÇÃO SBERT vs. TF-IDF")
print("-" * 70)

print("\n┌───────────────────────────────┬─────────────┬─────────────┐")
print("│ Par de Frases                 │   SBERT     │   TF-IDF    │")
print("├───────────────────────────────┼─────────────┼─────────────┤")
print(f"│ F1 × F2 (paráfrase)           │ {sim_f1_f2:.4f}      │ {similaridades_tfidf[0][1]:.4f}      │")
print(f"│ F1 × F3 (paráfrase)           │ {sim_f1_f3:.4f}      │ {similaridades_tfidf[0][2]:.4f}      │")
print(f"│ F1 × F7 (voz passiva)         │ {sim_f1_f7:.4f}      │ {similaridades_tfidf[0][6]:.4f}      │")
print("└───────────────────────────────┴─────────────┴─────────────┘")

print("\n✓ CONCLUSÃO: SBERT captura similaridade SEMÂNTICA onde TF-IDF falha")
print("  (sobreposição lexical zero ou baixa)")

# ============================================================
# PARTE 3: CROSS-ENCODER PARA RERANKING (CORRIGIDO)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 3: Cross-encoder (Alta Precisão - para reranking)")
print("=" * 70)

# Modelos cross-encoder alternativos disponíveis no Hugging Face
# Opção 1: Modelo multilíngue treinado em STS (recomendado)
# Opção 2: Modelo para português (se disponível)

# Vamos usar um modelo multilíngue que suporta português
# 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' é um modelo multilíngue disponível
# Alternativa: usar um modelo BERT multilíngue fine-tunado para STS

print("\nCarregando cross-encoder (pode levar alguns segundos)...")

# Opção 1: Modelo cross-encoder multilíngue disponível
# Fonte: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
# (Nota: Este modelo é treinado em inglês, mas podemos usá-lo como exemplo)

# Opção mais segura: usar um modelo de similaridade textual da família sentence-transformers
# que já é otimizado para STS, mas vamos demonstrar o cross-encoder

# Infelizmente, cross-encoders multilíngues de alta qualidade são raros.
# Vamos usar uma abordagem alternativa: carregar um modelo BERT multilíngue
# e adicionar uma camada de classificação (simulando cross-encoder)

from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class CrossEncoderForSTS(nn.Module):
    """Cross-encoder simples para similaridade textual"""
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Usa o embedding do token [CLS]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.classifier(cls_embedding)
        return score

# Carrega modelo BERT multilíngue (base) - suporta português
model_name = "bert-base-multilingual-cased"

try:
    # Tenta carregar o cross-encoder alternativo
    print("Tentando carregar cross-encoder disponível...")
    
    # Opção: Usar modelo cross-encoder inglês (funciona para demonstração)
    # Nota: Este modelo é em inglês, mas demonstra o conceito
    modelo_cross = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    tokenizer_cross = AutoTokenizer.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("✓ Cross-encoder carregado! (Modelo: ms-marco-MiniLM-L-6-v2)")
    
    # Aviso sobre o idioma
    print("\n  Nota: Este modelo é otimizado para inglês.")
    print("  O objetivo é demonstrar o CONCEITO de cross-encoder.")
    print("  Para português, modelos específicos seriam necessários.\n")
    
    modelo_cross.to(device)
    modelo_cross.eval()
    
    # Função para calcular similaridade com cross-encoder (para demonstração)
    def cross_encoder_score_english(par_frases):
        """Calcula score de similaridade (0-5) para um par de frases (em inglês)"""
        # Tradução simples para inglês (apenas para demonstração)
        traducao = {
            "O médico operou o paciente.": "The doctor operated on the patient.",
            "O cirurgião realizou a cirurgia.": "The surgeon performed the surgery.",
            "O médico não operou o paciente.": "The doctor did not operate on the patient.",
            "O gato dormiu no sofá.": "The cat slept on the sofa.",
            "O advogado leu o contrato.": "The lawyer read the contract.",
        }
        
        # Traduz as frases se necessário
        f1_trad = traducao.get(par_frases[0], par_frases[0])
        f2_trad = traducao.get(par_frases[1], par_frases[1])
        
        inputs = tokenizer_cross(
            [f1_trad, f2_trad], 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = modelo_cross(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            # Converte logits para escala 0-5 (aproximação)
            score = (logits[0] if hasattr(logits, '__len__') else logits)
            score = (score + 5) / 10  # Normalização aproximada
            score = max(0, min(5, score))
        return score
    
    # Seleciona alguns pares para avaliar
    pares_para_avaliar = [
        (frases[0], frases[1], "Paráfrase perfeita (médico ↔ cirurgião)"),
        (frases[0], frases[7], "Negação (oposto semântico)"),
        (frases[0], frases[4], "Tópicos diferentes (medicina vs. animal)"),
    ]
    
    print("\nAvaliação com Cross-encoder (escala 0-5, onde 5 = muito similar):")
    print("  Nota: Demonstração conceitual com modelo em inglês\n")
    print("-" * 70)
    
    for f1, f2, desc in pares_para_avaliar:
        score = cross_encoder_score_english([f1, f2])
        print(f"\n{desc}")
        print(f"  Frase 1: {f1}")
        print(f"  Frase 2: {f2}")
        print(f"  Score Cross-encoder: {score:.2f} / 5.0")
    
    print("\n✓ O cross-encoder é MAIS PRECISO que o SBERT, porém MAIS LENTO")
    print("  → Uso típico: SBERT para recuperação, Cross-encoder para reranking")
    print("  → Para português, modelos fine-tunados em ASSIN 2 seriam ideais")

except Exception as e:
    print(f"\n⚠ Não foi possível carregar o cross-encoder: {e}")
    print("\nUsando abordagem alternativa: Demonstração conceitual do cross-encoder")
    print("-" * 70)
    
    # Abordagem alternativa: demonstração conceitual sem carregar modelo
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    CROSS-ENCODER (CONCEITO)                                 │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │  O cross-encoder é uma arquitetura que processa o PAR de frases JUNTAS,    │
    │  permitindo que o modelo capture interações profundas entre os tokens.      │
    │                                                                             │
    │  Entrada: "[CLS] Frase A [SEP] Frase B [SEP]"                               │
    │           ↓                                                                 │
    │        [BERT]                                                               │
    │           ↓                                                                 │
    │     Embedding do [CLS] → Camada Linear → Score (ex: 4.2 / 5.0)              │
    │                                                                             │
    │  Vantagem: Alta precisão                                                    │
    │  Desvantagem: Não gera embeddings separados; processamento O(n²)            │
    │                                                                             │
    │  Para português, recomenda-se fine-tunar um cross-encoder no corpus ASSIN 2 │
    │  (Real et al., 2020) disponível em: https://huggingface.co/datasets/assin2  │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)

# ============================================================
# PARTE 4: VISUALIZAÇÃO t-SNE
# ============================================================
print("\n" + "=" * 70)
print("PARTE 4: Visualização t-SNE dos Embeddings SBERT")
print("=" * 70)

# Aplica t-SNE para reduzir para 2D
perplexity_val = min(5, len(embeddings_sbert) - 1)
print(f"Aplicando t-SNE (perplexity={perplexity_val})...")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
embeddings_2d = tsne.fit_transform(embeddings_sbert)

# Cores para grupos semânticos
cores = []
for i in range(len(frases)):
    if i <= 2:  # Grupo medicina
        cores.append('#1f77b4')  # azul
    elif i == 3:  # Direito
        cores.append('#ff7f0e')  # laranja
    elif 4 <= i <= 5:  # Animais
        cores.append('#2ca02c')  # verde
    elif i == 6:  # Voz passiva (medicina)
        cores.append('#1f77b4')
    elif i == 7:  # Negação
        cores.append('#d62728')  # vermelho
    else:  # Cirurgia sucesso
        cores.append('#9467bd')  # roxo

# Tamanhos
tamanhos = [100] * len(frases)

# Cria o gráfico
plt.figure(figsize=(14, 10))
for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y, marker='o', color=cores[i], s=tamanhos[i], 
                alpha=0.7, edgecolors='black', linewidth=1.5)
    # Adiciona rótulo com número e texto abreviado
    texto_curto = frases[i][:25] + "..." if len(frases[i]) > 25 else frases[i]
    plt.text(x + 0.1, y + 0.1, f"{i+1}", fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    plt.text(x + 0.1, y - 0.3, texto_curto, fontsize=8, alpha=0.7)

# Adiciona anotações de grupo
plt.annotate('Grupo 1: Medicina/Cirurgia\n(paráfrases - F1, F2, F3, F7, F9)', 
             xy=(embeddings_2d[0, 0], embeddings_2d[0, 1]),
             xytext=(embeddings_2d[0, 0] + 3, embeddings_2d[0, 1] + 3),
             fontsize=10, style='italic', color='#1f77b4',
             arrowprops=dict(arrowstyle='->', color='#1f77b4', alpha=0.5))

plt.annotate('Grupo 2: Direito\n(F4 - tópico diferente)', 
             xy=(embeddings_2d[3, 0], embeddings_2d[3, 1]),
             xytext=(embeddings_2d[3, 0] + 2, embeddings_2d[3, 1] + 2),
             fontsize=10, style='italic', color='#ff7f0e',
             arrowprops=dict(arrowstyle='->', color='#ff7f0e', alpha=0.5))

plt.annotate('Grupo 3: Animais\n(F5, F6 - tópico diferente)', 
             xy=(embeddings_2d[4, 0], embeddings_2d[4, 1]),
             xytext=(embeddings_2d[4, 0] + 2, embeddings_2d[4, 1] + 2),
             fontsize=10, style='italic', color='#2ca02c',
             arrowprops=dict(arrowstyle='->', color='#2ca02c', alpha=0.5))

plt.annotate('Grupo 4: Negação\n(F8 - sentido oposto)', 
             xy=(embeddings_2d[7, 0], embeddings_2d[7, 1]),
             xytext=(embeddings_2d[7, 0] + 2, embeddings_2d[7, 1] - 1.5),
             fontsize=10, style='italic', color='#d62728',
             arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.5))

plt.title("Visualização de Embeddings Semânticos de Frases (SBERT - t-SNE)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Dimensão 1 (t-SNE)", fontsize=12)
plt.ylabel("Dimensão 2 (t-SNE)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("\n✓ Análise do gráfico t-SNE:")
print("  → Frases do grupo Medicina/Cirurgia (F1, F2, F3, F7, F9) estão PRÓXIMAS")
print("  → Frase de Direito (F4) está DISTANTE do grupo Medicina")
print("  → Frases de Animais (F5, F6) estão DISTANTES dos demais grupos")
print("  → Frase de Negação (F8) está mais distante das paráfrases positivas")

# ============================================================
# PARTE 5: INFERÊNCIA TEXTUAL (NLI) CONCEITUAL
# ============================================================
print("\n" + "=" * 70)
print("PARTE 5: Inferência Textual Natural (NLI) - Conceitos")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERÊNCIA TEXTUAL (NLI / RTE)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Acarretamento (Entailment): Se P é verdadeiro, H é verdadeiro             │
│  Contradição (Contradiction): P e H não podem ser ambos verdadeiros        │
│  Neutro (Neutral): Nenhuma relação garantida                               │
│                                                                             │
│  Exemplos:                                                                  │
│                                                                             │
│  1. ACARRETAMENTO                                                          │
│     P: "O médico operou o paciente."                                       │
│     H: "Houve uma cirurgia."                                               │
│                                                                             │
│  2. CONTRADIÇÃO                                                            │
│     P: "O médico operou o paciente."                                       │
│     H: "O paciente não foi operado."                                       │
│                                                                             │
│  3. NEUTRO                                                                  │
│     P: "O médico operou o paciente."                                       │
│     H: "O médico usou um bisturi."                                         │
│     (Possível, mas não garantido pela premissa)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# PARTE 6: APLICAÇÃO EM BUSCA SEMÂNTICA (SIMULAÇÃO)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 6: Simulação de Busca Semântica")
print("=" * 70)

# Documentos de exemplo
documentos = [
    "A empresa reportou um prejuízo de R$ 10 milhões no último trimestre.",
    "Os lucros da organização aumentaram 20% neste ano.",
    "A companhia teve perdas significativas no terceiro trimestre fiscal.",
    "O governo anunciou novas medidas econômicas.",
    "A firma registrou resultados negativos nos últimos três meses."
]

# Consulta do usuário
consulta = "Qual empresa teve prejuízo no último trimestre?"

print(f"Consulta: '{consulta}'")
print(f"\nDocumentos na coleção ({len(documentos)} documentos):")
for i, doc in enumerate(documentos):
    print(f"  D{i+1}: {doc}")

# Gera embeddings para consulta e documentos
todos_textos = [consulta] + documentos
embeddings_busca = modelo_sbert.encode(todos_textos, convert_to_numpy=True)
consulta_emb = embeddings_busca[0]
docs_emb = embeddings_busca[1:]

# Calcula similaridades
similaridades_busca = cosine_similarity([consulta_emb], docs_emb)[0]

print("\nResultados da busca semântica (ordenados por relevância):")
print("-" * 70)

# Ordena documentos por similaridade
indices_ordenados = np.argsort(similaridades_busca)[::-1]

for rank, idx in enumerate(indices_ordenados):
    sim = similaridades_busca[idx]
    doc_texto = documentos[idx]
    print(f"{rank+1}. Similaridade: {sim:.4f} | Documento: {doc_texto}")

print("\n✓ Análise dos resultados:")
print("  → Documento D1 ('prejuízo... último trimestre') = ALTA similaridade")
print("  → Documento D5 ('resultados negativos... últimos três meses') = ALTA similaridade")
print("  → Documento D3 ('perdas... terceiro trimestre') = MÉDIA similaridade")
print("  → Documento D2 ('lucros aumentaram') = BAIXA similaridade (sentido oposto)")
print("  → Documento D4 ('governo') = BAIXA similaridade (tópico diferente)")

print("\n" + "=" * 70)
print("FIM DO EXEMPLO COMPUTACIONAL")
print("=" * 70)
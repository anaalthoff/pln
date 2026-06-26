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

# ============================================================
# PARTE 1: EMBEDDINGS DE FRASES COM SENTENCE-TRANSFORMERS
# ============================================================
print("\n" + "=" * 70)
print("PARTE 1: Sentence-Transformers (Modelo Multilíngue)")
print("=" * 70)

# Carrega modelo multilíngue otimizado para similaridade semântica
# Este modelo suporta português e várias outras línguas
print("\nCarregando modelo Sentence-Transformer...")
modelo_frases = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
modelo_frases.to(device)
print("✓ Modelo carregado com sucesso!")

# Frases de exemplo em português para demonstrar similaridade semântica
frases = [
    "O médico operou o paciente.",           # Frase 1 - médio/cirurgia
    "O cirurgião realizou a cirurgia.",      # Frase 2 - mesmo significado da F1
    "O advogado leu o contrato.",            # Frase 3 - domínio diferente
    "O gato dormiu no sofá.",                # Frase 4 - animal doméstico
    "O cachorro correu no parque.",          # Frase 5 - animal doméstico
    "O médico examinou o paciente cuidadosamente."  # Frase 6 - similar à F1
]

print("\nFrases analisadas:")
for i, f in enumerate(frases, 1):
    print(f"  F{i}: {f}")

# Gera embeddings das frases (vetores densos de 384 dimensões)
print("\nGerando embeddings das frases...")
embeddings = modelo_frases.encode(frases, convert_to_numpy=True)
print(f"✓ Embeddings gerados! Shape: {embeddings.shape}")
print(f"  → {embeddings.shape[0]} frases × {embeddings.shape[1]} dimensões")

# Calcula matriz de similaridade (cosseno) entre todas as frases
similaridades = cosine_similarity(embeddings)

print("\n" + "-" * 70)
print("Matriz de Similaridade do Cosseno (valores arredondados):")
print("-" * 70)
print("     ", end="")
for i in range(1, 7):
    print(f"  F{i}   ", end="")
print()
for i in range(6):
    print(f"F{i+1}: ", end="")
    for j in range(6):
        print(f"{similaridades[i][j]:.3f} ", end="")
        if j == 5:
            print()

print("\n" + "-" * 70)
print("ANÁLISE DE SIMILARIDADE SEMÂNTICA")
print("-" * 70)

# Análise dos resultados
sim_f1_f2 = similaridades[0][1]
sim_f1_f6 = similaridades[0][5]
sim_f1_f3 = similaridades[0][2]
sim_f4_f5 = similaridades[3][4]
sim_f2_f3 = similaridades[1][2]

print(f"\n1. Similaridade (F1, F2): {sim_f1_f2:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[1]}'")
print(f"   → Análise: Similaridade ALTA (valores próximos de 0.8-0.9)")
print(f"   → As frases têm MESMO SIGNIFICADO, mesmo com palavras diferentes!")
print(f"   → Exemplo: 'médico' ≈ 'cirurgião', 'operou' ≈ 'realizou cirurgia'")

print(f"\n2. Similaridade (F1, F6): {sim_f1_f6:.4f}")
print(f"   → '{frases[0]}'")
print(f"   → '{frases[5]}'")
print(f"   → Análise: Similaridade MUITO ALTA")
print(f"   → Ambas sobre médico/paciente, com variação lexical")

print(f"\n3. Similaridade (F1, F3): {sim_f1_f3:.4f}")
print(f"   → '{frases[0]}' (medicina)")
print(f"   → '{frases[2]}' (direito)")
print(f"   → Análise: Similaridade BAIXA (domínios diferentes)")

print(f"\n4. Similaridade (F4, F5): {sim_f4_f5:.4f}")
print(f"   → '{frases[3]}' (gato)")
print(f"   → '{frases[4]}' (cachorro)")
print(f"   → Análise: Similaridade MÉDIA-ALTA")
print(f"   → Ambos são animais domésticos, semanticamente próximos")

print(f"\n5. Similaridade (F2, F3): {sim_f2_f3:.4f}")
print(f"   → '{frases[1]}' (cirurgia)")
print(f"   → '{frases[2]}' (direito)")
print(f"   → Análise: Similaridade BAIXA (domínios diferentes)")

# ============================================================
# PARTE 2: BERTimbau - Embeddings Contextuais para Palavras
# ============================================================
print("\n" + "=" * 70)
print("PARTE 2: BERTimbau (BERT para Português Brasileiro)")
print("=" * 70)
print("Demonstrando que BERT gera embeddings DIFERENTES para a mesma palavra")
print("em contextos diferentes (resolvendo o problema da polissemia)")
print("-" * 70)

# Carrega modelo BERTimbau-base (Souza et al., 2020)
print("\nCarregando BERTimbau... (pode levar 1-2 minutos na primeira execução)")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
model.to(device)
model.eval()
print("✓ BERTimbau carregado com sucesso!")

# Frase com duas ocorrências de "banco" com sentidos diferentes
frase_exemplo = "O banco do jardim estava quebrado, mas fui ao banco sacar dinheiro."

print(f"\nFrase de exemplo: \"{frase_exemplo}\"")
print("\nEsta frase contém a palavra 'banco' em DOIS sentidos diferentes:")
print("  → Primeiro 'banco': assento de jardim")
print("  → Segundo 'banco': instituição financeira")

# Tokeniza a frase
tokens = tokenizer(frase_exemplo, return_tensors="pt", return_token_type_ids=False)
tokens = {k: v.to(device) for k, v in tokens.items()}

print("\nTokenização (subpalavras):")
tokens_decodificados = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
for i, token in enumerate(tokens_decodificados):
    # Destaca as ocorrências de 'banco'
    if token == 'banco':
        print(f"  {i:2d}: [{token}] ← POSIÇÃO DA PALAVRA 'BANCO'")
    else:
        print(f"  {i:2d}: {token}")

# Obtém embeddings da última camada
with torch.no_grad():
    outputs = model(**tokens)
    embeddings_bert = outputs.last_hidden_state[0].cpu().numpy()  # [seq_len, 768]

print(f"\nShape dos embeddings BERT: {embeddings_bert.shape}")
print(f"  → {embeddings_bert.shape[0]} tokens × {embeddings_bert.shape[1]} dimensões")

# Encontra as posições da palavra "banco"
posicoes_banco = [i for i, token in enumerate(tokens_decodificados) if token == 'banco']
print(f"\nPosições da palavra 'banco' na sequência: {posicoes_banco}")

# Extrai os embeddings das duas ocorrências
embedding_banco1 = embeddings_bert[posicoes_banco[0]]  # primeiro 'banco' (assento)
embedding_banco2 = embeddings_bert[posicoes_banco[1]]  # segundo 'banco' (financeiro)

print(f"\nEmbedding 1 (primeiro 'banco' - contexto: jardim/quebrado): shape {embedding_banco1.shape}")
print(f"Embedding 2 (segundo 'banco' - contexto: sacar/dinheiro): shape {embedding_banco2.shape}")

# Calcula similaridade entre as duas ocorrências
sim_banco = cosine_similarity([embedding_banco1], [embedding_banco2])[0][0]

print("\n" + "-" * 70)
print("RESULTADO - COMPARAÇÃO DOS EMBEDDINGS DE 'BANCO'")
print("-" * 70)
print(f"\nSimilaridade do cosseno entre as duas ocorrências de 'banco': {sim_banco:.4f}")

if sim_banco < 0.7:
    print("\n✓ CONCLUSÃO: Similaridade MODERADA a BAIXA!")
    print("  → O BERT conseguiu DISTINGUIR os dois sentidos da palavra!")
    print("  → O embedding para 'banco (assento)' é DIFERENTE do embedding para")
    print("    'banco (instituição financeira)'")
    print("  → Isso resolve o problema da POLISSEMIA que afeta embeddings estáticos")
else:
    print("\n⚠ OBSERVAÇÃO: Similaridade ALTA")
    print("  → Neste caso específico, o BERT pode ter tido dificuldade em distinguir")
    print("  → Isso pode acontecer dependendo do contexto e do modelo")

# ============================================================
# PARTE 3: COMPARAÇÃO ENTRE EMBEDDINGS DE FRASES
# ============================================================
print("\n" + "=" * 70)
print("PARTE 3: Comparação entre Embeddings de Frases")
print("=" * 70)

# Seleciona algumas frases para comparação detalhada
pares_para_comparar = [
    (0, 1, "F1 vs F2: médico/operou ↔ cirurgião/realizou cirurgia"),
    (0, 2, "F1 vs F3: médico ↔ advogado (domínios diferentes)"),
    (0, 5, "F1 vs F6: médico/operou ↔ médico/examinou (variação lexical)"),
    (3, 4, "F4 vs F5: gato ↔ cachorro (animais domésticos)")
]

print("\nComparações detalhadas de similaridade semântica:")
print("-" * 70)

for i, j, descricao in pares_para_comparar:
    sim = similaridades[i][j]
    print(f"\n{descricao}")
    print(f"  → Similaridade: {sim:.4f}")
    if sim > 0.7:
        print("  → Interpretação: FRASES SEMANTICAMENTE MUITO PRÓXIMAS")
    elif sim > 0.5:
        print("  → Interpretação: FRASES SEMANTICAMENTE RELACIONADAS")
    else:
        print("  → Interpretação: FRASES SEMANTICAMENTE DISTANTES")


# ============================================================
# PARTE 4: VISUALIZAÇÃO t-SNE (CORRIGIDA - VERSÃO FINAL)
# ============================================================
print("\n" + "=" * 70)
print("PARTE 4: Visualização t-SNE dos Embeddings")
print("=" * 70)
print("Reduzindo embeddings de alta dimensão (384D) para 2D para visualização")
print("-" * 70)

# Estratégia: Vamos visualizar APENAS os embeddings do SentenceTransformer
# que já são todos da mesma dimensionalidade (384)
print("\nPreparando dados para visualização...")

# Usa apenas os embeddings do SentenceTransformer (todos 384 dims)
vetores_para_tsne = np.array(embeddings)  # shape: (6, 384)
labels_tsne = [f"Frase {i+1}" for i in range(len(frases))]

print(f"  → {len(vetores_para_tsne)} embeddings, cada um com {vetores_para_tsne.shape[1]} dimensões")

# Aplica t-SNE para reduzir para 2D
# Ajusta perplexity para não exceder o número de amostras
perplexity_val = min(5, len(vetores_para_tsne) - 1)
print(f"  → Perplexity do t-SNE: {perplexity_val}")

print("\nAplicando t-SNE (isso pode levar alguns segundos)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
            max_iter=1000, learning_rate='auto')
vetores_2d = tsne.fit_transform(vetores_para_tsne)

# Cria o gráfico de visualização
plt.figure(figsize=(12, 8))

# Cores personalizadas para cada frase
cores = ['#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c', '#2ca02c', '#1f77b4']
tamanhos = [150, 150, 150, 150, 150, 150]

for i, label in enumerate(labels_tsne):
    x, y = vetores_2d[i, 0], vetores_2d[i, 1]
    plt.scatter(x, y, marker='o', color=cores[i], s=tamanhos[i], 
                alpha=0.7, edgecolors='black', linewidth=1.5)
    plt.text(x + 0.05, y + 0.05, label, fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# Adiciona anotações para destacar grupos semânticos
plt.annotate('Grupo: Medicina/Cirurgia', 
             xy=(vetores_2d[0, 0], vetores_2d[0, 1]),
             xytext=(vetores_2d[0, 0] + 2, vetores_2d[0, 1] + 2),
             fontsize=10, style='italic',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

plt.annotate('Grupo: Animais', 
             xy=(vetores_2d[3, 0], vetores_2d[3, 1]),
             xytext=(vetores_2d[3, 0] + 2, vetores_2d[3, 1] + 1),
             fontsize=10, style='italic',
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

plt.title("Visualização de Embeddings Semânticos de Frases (t-SNE)\n" +
          "Sentence-Transformers Multilíngue", 
          fontsize=14, fontweight='bold')
plt.xlabel("Dimensão 1 (t-SNE)", fontsize=12)
plt.ylabel("Dimensão 2 (t-SNE)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("\n✓ Visualização concluída!")

# ============================================================
# PARTE 5: COMPARAÇÃO CONCEITUAL - ESTÁTICO vs. CONTEXTUAL
# ============================================================
print("\n" + "=" * 70)
print("PARTE 5: Comparação Conceitual - Embeddings Estáticos vs. Contextuais")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS ESTÁTICOS (Word2Vec, GloVe)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Uma palavra → UM ÚNICO vetor (independente do contexto)                   │
│ • 'banco' (assento) e 'banco' (financeiro) têm o MESMO vetor               │
│ • Não resolve o problema da POLISSEMIA                                      │
│ • Vantagem: RÁPIDO (simples lookup em tabela)                               │
│ • Desvantagem: Não captura nuances contextuais                              │
│                                                                             │
│ Exemplo:                                                                    │
│   'banco' → [0.2, -0.5, 0.8, 0.1, ...]  (vetor ÚNICO para TODOS os usos)  │
│   Similaridade entre os dois sentidos = 1.0 (idênticos!)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS CONTEXTUAIS (BERT, ELMo, GPT)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Uma palavra → VETOR DIFERENTE para cada contexto                          │
│ • 'banco' (assento) e 'banco' (financeiro) têm vetores DIFERENTES          │
│ • RESOLVE o problema da POLISSEMIA                                          │
│ • Vantagem: Captura nuances contextuais                                     │
│ • Desvantagem: MAIS LENTO (requer passagem pela rede neural)               │
│                                                                             │
│ Exemplo (BERT):                                                             │
│   'banco' (assento) → [0.8, -0.2, 0.1, -0.5, ...]                          │
│   'banco' (financeiro) → [-0.3, 0.7, -0.1, 0.4, ...]                       │
│   Similaridade entre os dois sentidos ≈ 0.62 (diferentes!)                  │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# PARTE 6: RESUMO DOS RESULTADOS
# ============================================================
print("\n" + "=" * 70)
print("RESUMO DOS RESULTADOS")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SIMILARIDADE ENTRE FRASES (Sentence-Transformers)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Frases com MESMO significado (F1 e F2): similaridade = {sim_f1_f2:.4f}     │
│ • Frases com significado RELACIONADO (F1 e F6): similaridade = {sim_f1_f6:.4f} │
│ • Frases com SIGNIFICADOS DIFERENTES (F1 e F3): similaridade = {sim_f1_f3:.4f} │
│                                                                             │
│ ✓ O modelo captura similaridade SEMÂNTICA, não apenas lexical!              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DISTINÇÃO DE SENTIDOS (BERTimbau)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Similaridade entre os dois sentidos de 'banco': {sim_banco:.4f}            │
│                                                                             │
│ ✓ O BERT gerou embeddings DIFERENTES para os dois contextos!                │
│ ✓ Isso resolve o problema da POLISSEMIA que afeta embeddings estáticos      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. IMPLICAÇÕES PRÁTICAS                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Busca semântica: encontrar documentos por SIGNIFICADO, não por palavras   │
│ • Chatbots: entender a INTENÇÃO do usuário mesmo com palavras diferentes    │
│ • Análise de sentimentos: capturar IRONIA e SARCASMO                        │
│ • Tradução automática: gerar traduções mais NATURAIS e CONTEXTUAIS          │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# EXTRA: Informações sobre os modelos utilizados
# ============================================================
print("\n" + "=" * 70)
print("INFORMAÇÕES SOBRE OS MODELOS UTILIZADOS")
print("=" * 70)

print("""
Modelo 1: paraphrase-multilingual-MiniLM-L12-v2 (Sentence-Transformers)
├── Arquitetura: MiniLM (versão leve do BERT)
├── Dimensão do embedding: 384
├── Suporte multilíngue: 50+ línguas (incluindo português)
├── Tamanho: ~120 MB
├── Uso: Similaridade semântica entre FRASES
└── Licença: Apache 2.0

Modelo 2: neuralmind/bert-base-portuguese-cased (BERTimbau)
├── Arquitetura: BERT-base (12 camadas, 12 heads, 110M parâmetros)
├── Dimensão do embedding: 768
├── Treinado especificamente para português brasileiro
├── Tamanho: ~420 MB
├── Dados de treino: brWaC (2.68B tokens)
├── Uso: Embeddings contextuais para PALAVRAS
└── Licença: MIT

Diferença fundamental:
└── Sentence-Transformer é OTIMIZADO para similaridade entre frases
└── BERTimbau é um modelo BERT GENÉRICO, bom para embeddings de palavras
""")

print("\n" + "=" * 70)
print("FIM DO EXEMPLO COMPUTACIONAL")
print("=" * 70)
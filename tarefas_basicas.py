# Tarefas fundamentais que o spaCy realiza automaticamente ao processar um texto
import spacy
from spacy import displacy
import pandas as pd

# Carregar o modelo de linguagem para português
nlp_pt = spacy.load("pt_core_news_sm")

# Texto de exemplo
texto = "O gerente disse ao funcionário que seu projeto foi excelente, mas o bônus foi para o time todo."

# Processar o texto (a mágica do spaCy acontece aqui)
doc_pt = nlp_pt(texto)

# 1. Tokenização
print("=" * 60)
print("1. TOKENIZAÇÃO")
print("=" * 60)
tokens = [token.text for token in doc_pt]
print(f"Texto original: {texto}")
print(f"Tokens: {tokens}")
print(f"Número de tokens: {len(doc_pt)}")

# 2. Análise Morfossintática (POS Tagging)
print("\n" + "=" * 60)
print("2. ANÁLISE MORFOSSINTÁTICA (POS TAGGING)")
print("=" * 60)
print(f"{'Token':<15} {'POS Tag':<15} {'POS (descrição)':<25} {'Dependência':<15}")
print("-" * 70)
for token in doc_pt:
    print(f"{token.text:<15} {token.pos_:<15} {spacy.explain(token.pos_):<25} {token.dep_:<15}")

# 3. Lematização (redução à forma base/dicionário)
print("\n" + "=" * 60)
print("3. LEMATIZAÇÃO")
print("=" * 60)
lemas = [(token.text, token.lemma_) for token in doc_pt if not token.is_punct]
df_lemas = pd.DataFrame(lemas, columns=["Palavra Original", "Lema"])
print(df_lemas.to_string(index=False))

# O NER identifica e classifica entidades como pessoas, organizações, locais, datas, etc.
# Texto com entidades nomeadas
texto_ner = "A Google anunciou hoje que João Silva, funcionário da Microsoft em São Paulo, receberá um bônus de R$ 10.000 em 15/12/2024."

doc_ner = nlp_pt(texto_ner)

print("=" * 60)
print("4. RECONHECIMENTO DE ENTIDADES NOMEADAS (NER)")
print("=" * 60)

entidades = []
for ent in doc_ner.ents:
    entidades.append({
        "Entidade": ent.text,
        "Rótulo": ent.label_,
        "Descrição": spacy.explain(ent.label_)
    })

df_entidades = pd.DataFrame(entidades)
print(df_entidades.to_string(index=False))

# Visualização interativa (funciona em Jupyter Notebook)
# displacy.render(doc_ner, style="ent", jupyter=True)

# Visualização textual
print("\nVisualização textual das entidades:")
for ent in doc_ner.ents:
    print(f"  - '{ent.text}' → {ent.label_} ({spacy.explain(ent.label_)})")

# O spaCy também permite acesso a embeddings de palavras, que são representações vetoriais que capturam significado contextual
import numpy as np

# Carregar modelo de português com vetores
nlp_pt = spacy.load("pt_core_news_lg")  # modelo grande com vetores

# Palavra ambígua em português
print("=" * 60)
print("DESAMBIGUAÇÃO CONTEXTUAL: A palavra 'MANGA'")
print("=" * 60)

def similaridade_contextual(frase, palavra_alvo, sentido_referencia):
    """Calcula similaridade entre o contexto da frase e um sentido de referência."""
    doc = nlp_pt(frase)
    
    # Encontrar o token da palavra alvo
    token_alvo = None
    for token in doc:
        if token.text.lower() == palavra_alvo:
            token_alvo = token
            break
    
    if token_alvo is None:
        return 0.0
    
    # Calcular vetor médio do contexto (excluindo a palavra alvo)
    vetores_contexto = []
    for token in doc:
        if token.text.lower() != palavra_alvo and token.has_vector:
            vetores_contexto.append(token.vector)
    
    if not vetores_contexto:
        return 0.0
    
    vetor_contexto = np.mean(vetores_contexto, axis=0)
    
    # Criar documento do sentido de referência
    doc_referencia = nlp_pt(sentido_referencia)
    
    # Calcular similaridade por cosseno
    similaridade = token_alvo.similarity(doc_referencia)
    return similaridade

# Testando diferentes contextos
contextos = [
    ("Comi uma manga suculenta no almoço", "fruta"),
    ("A costureira usou a manga da camisa", "parte_roupa"),
    ("O desenho tem traços de estilo manga", "estilo_quadrinho")
]

for frase, sentido in contextos:
    sim = similaridade_contextual(frase, "manga", sentido)
    print(f"\nFrase: '{frase}'")
    print(f"  → Sentido esperado: {sentido}")
    print(f"  → Similaridade: {sim:.4f}")
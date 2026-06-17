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
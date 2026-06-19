# Pipeline de Pré-processamento para Busca
import re
from typing import List, Tuple
import spacy
from spacy import displacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BuscaPLN:
    """Sistema simples de busca com pré-processamento usando spaCy."""
    
    def __init__(self, modelo_idioma="pt_core_news_sm"):
        self.nlp = spacy.load(modelo_idioma)
        self.documentos = []
        self.embeddings_documentos = []
    
    def preprocessar_texto(self, texto: str) -> spacy.tokens.Doc:
        """Aplica pipeline completo de PLN ao texto."""
        # O spaCy já faz tudo: tokenização, lematização, remoção de stopwords, etc.
        doc = self.nlp(texto.lower())
        
        # Filtrar tokens: remover pontuação, espaços e stopwords (opcional)
        tokens_filtrados = [
            token.lemma_ for token in doc 
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        
        return tokens_filtrados
    
    def adicionar_documento(self, texto: str, metadados: dict = None):
        """Adiciona um documento ao índice de busca."""
        doc_processado = self.preprocessar_texto(texto)
        
        # Calcular embedding do documento (média dos vetores dos tokens)
        doc_nlp = self.nlp(texto)
        if doc_nlp.has_vector:
            embedding = doc_nlp.vector
        else:
            embedding = None
        
        self.documentos.append({
            "texto_original": texto,
            "tokens_processados": doc_processado,
            "embedding": embedding,
            "metadados": metadados or {}
        })
        
        if embedding is not None:
            self.embeddings_documentos.append(embedding)
    
    def buscar(self, consulta: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Busca documentos similares à consulta."""
        # Processar consulta
        consulta_processada = self.preprocessar_texto(consulta)
        doc_consulta = self.nlp(consulta)
        
        if not doc_consulta.has_vector or not self.embeddings_documentos:
            return []
        
        # Calcular similaridades
        vetor_consulta = doc_consulta.vector
        similaridades = cosine_similarity([vetor_consulta], self.embeddings_documentos)[0]
        
        # Ordenar e retornar top_k
        indices_ordenados = np.argsort(similaridades)[::-1][:top_k]
        
        resultados = []
        for idx in indices_ordenados:
            if similaridades[idx] > 0:  # Ignorar similaridade zero
                resultados.append((self.documentos[idx], similaridades[idx]))
        
        return resultados
    
    def buscar_por_palavras_chave(self, consulta: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Busca baseada em palavras-chave (abordagem tradicional)."""
        termos_busca = set(self.preprocessar_texto(consulta))
        
        resultados = []
        for doc in self.documentos:
            # Contar quantos termos da consulta aparecem no documento
            tokens_doc = set(doc["tokens_processados"])
            interseccao = termos_busca.intersection(tokens_doc)
            score = len(interseccao) / len(termos_busca) if termos_busca else 0
            resultados.append((doc, score))
        
        # Ordenar por score decrescente
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]
    
# ===== DEMONSTRAÇÃO DO SISTEMA DE BUSCA =====

print("=" * 60)
print("8. SISTEMA DE BUSCA COMPLETO COM PRÉ-PROCESSAMENTO")
print("=" * 60)

# Criar sistema de busca
buscador = BuscaPLN(modelo_idioma="pt_core_news_sm")

# Adicionar documentos (notícias fictícias)
documentos_busca = [
    "Google anuncia novo centro de pesquisa em inteligência artificial em São Paulo",
    "Microsoft lança atualização do Windows com foco em segurança e privacidade",
    "Apple apresenta novo iPhone com câmera de alta resolução e bateria de longa duração",
    "Pesquisadores brasileiros desenvolvem algoritmo inovador para processamento de linguagem natural",
    "Amazon expande operações de logística no Brasil com novos centros de distribuição",
    "Universidade de São Paulo oferece curso gratuito sobre PLN e machine learning"
]

for i, doc in enumerate(documentos_busca):
    buscador.adicionar_documento(doc, {"id": i, "fonte": "Notícias Tech"})

# Realizar buscas
consultas = [
    "inteligência artificial no Brasil",
    "novidades da Apple",
    "curso de PLN gratuito"
]

for consulta in consultas:
    print(f"\n--- Busca: '{consulta}' ---")
    
    # Busca semântica (com embeddings)
    resultados_semanticos = buscador.buscar(consulta, top_k=2)
    print("\n🔍 Busca Semântica (Embeddings):")
    for doc, score in resultados_semanticos:
        print(f"  Score: {score:.4f} | {doc['texto_original']}")
    
    # Busca por palavras-chave (tradicional)
    resultados_keywords = buscador.buscar_por_palavras_chave(consulta, top_k=2)
    print("\n🔎 Busca por Palavras-chave:")
    for doc, score in resultados_keywords:
        print(f"  Score: {score:.4f} | {doc['texto_original']}")
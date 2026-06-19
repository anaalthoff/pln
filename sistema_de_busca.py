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
    
print("Deu certo")
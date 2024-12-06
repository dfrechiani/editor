# text_analysis.py
import nltk
import spacy
import textstat
from typing import Dict, Any, List
from collections import Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize spaCy
try:
    nlp = spacy.load('pt_core_news_lg')
except OSError:
    logging.warning("Downloading Portuguese language model...")
    spacy.cli.download('pt_core_news_lg')
    nlp = spacy.load('pt_core_news_lg')

class TextAnalyzer:
    """
    Implementação de análise textual inspirada no Coh-Metrix,
    usando bibliotecas Python modernas para processamento de linguagem natural.
    """
    
    def __init__(self):
        self.nlp = nlp
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words=spacy.lang.pt.stop_words.STOP_WORDS
        )
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Realiza análise completa do texto, retornando métricas similares ao Coh-Metrix.
        """
        # Processamento básico
        doc = self.nlp(text)
        sentences = list(doc.sents)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        
        # Análises básicas
        basic_metrics = {
            "Word Count": len(words),
            "Sentence Count": len(sentences),
            "Paragraph Count": len(paragraphs),
            "Words per Sentence": round(len(words) / len(sentences), 2) if sentences else 0,
            "Sentences per Paragraph": round(len(sentences) / len(paragraphs), 2) if paragraphs else 0,
        }
        
        # Análise léxica
        lexical_metrics = self._analyze_lexical_features(doc, words)
        
        # Análise sintática
        syntactic_metrics = self._analyze_syntactic_features(doc)
        
        # Análise de coesão
        cohesion_metrics = self._analyze_cohesion_features(doc, sentences)
        
        # Combine todas as métricas
        return {
            **basic_metrics,
            **lexical_metrics,
            **syntactic_metrics,
            **cohesion_metrics
        }
    
    def _analyze_lexical_features(self, doc: spacy.tokens.Doc, words: List[str]) -> Dict[str, Any]:
        """Análise detalhada das características léxicas do texto."""
        # Contagem de palavras únicas
        unique_words = len(set(words))
        
        # Diversidade léxica
        lexical_diversity = round(unique_words / len(words) * 100, 2) if words else 0
        
        # Frequência de classes gramaticais
        pos_counts = Counter(token.pos_ for token in doc)
        
        # Densidade lexical (proporção de palavras de conteúdo)
        content_words = sum(1 for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'])
        lexical_density = round(content_words / len(words) * 100, 2) if words else 0
        
        return {
            "Unique Words": unique_words,
            "Lexical Diversity": lexical_diversity,
            "Lexical Density": lexical_density,
            "Nouns": pos_counts['NOUN'],
            "Verbs": pos_counts['VERB'],
            "Adjectives": pos_counts['ADJ'],
            "Adverbs": pos_counts['ADV']
        }
    
    def _analyze_syntactic_features(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        """Análise das características sintáticas do texto."""
        # Contagem de frases nominais e verbais
        noun_phrases = len(list(doc.noun_chunks))
        verb_phrases = sum(1 for token in doc if token.pos_ == 'VERB' and len(list(token.children)) > 0)
        
        # Complexidade sintática
        dependency_depths = [self._get_dependency_depth(token) for token in doc]
        avg_dependency_depth = round(sum(dependency_depths) / len(dependency_depths), 2) if dependency_depths else 0
        
        return {
            "Noun Phrases": noun_phrases,
            "Verb Phrases": verb_phrases,
            "Syntactic Complexity": avg_dependency_depth,
            "Complex Sentences": self._count_complex_sentences(doc)
        }
    
    def _analyze_cohesion_features(self, doc: spacy.tokens.Doc, sentences: List[spacy.tokens.Span]) -> Dict[str, Any]:
        """Análise das características de coesão do texto."""
        # Identificar conectivos
        connectives = self._identify_connectives(doc)
        
        # Análise de referências e repetições
        references = self._analyze_references(doc)
        
        # Similaridade entre sentenças adjacentes
        sentence_similarities = self._calculate_sentence_similarities(sentences)
        
        return {
            "Connectives": len(connectives),
            "Connective Types": {
                "Additive": len([c for c in connectives if c['type'] == 'additive']),
                "Causal": len([c for c in connectives if c['type'] == 'causal']),
                "Temporal": len([c for c in connectives if c['type'] == 'temporal']),
                "Contrastive": len([c for c in connectives if c['type'] == 'contrastive'])
            },
            "Reference Cohesion": references['score'],
            "Sentence Similarity": sentence_similarities
        }
    
    def _get_dependency_depth(self, token: spacy.tokens.Token) -> int:
        """Calcula a profundidade da árvore de dependência para um token."""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth
    
    def _count_complex_sentences(self, doc: spacy.tokens.Doc) -> int:
        """Conta o número de sentenças complexas (com mais de uma cláusula)."""
        return sum(1 for sent in doc.sents if len([t for t in sent if t.dep_ in ['ccomp', 'xcomp', 'advcl']]) > 0)
    
    def _identify_connectives(self, doc: spacy.tokens.Doc) -> List[Dict[str, str]]:
        """Identifica e classifica os conectivos no texto."""
        connective_patterns = {
            'additive': ['além disso', 'também', 'ademais', 'inclusive'],
            'causal': ['porque', 'pois', 'portanto', 'consequentemente'],
            'temporal': ['quando', 'depois', 'antes', 'enquanto'],
            'contrastive': ['mas', 'porém', 'entretanto', 'contudo']
        }
        
        connectives = []
        text = doc.text.lower()
        
        for conn_type, patterns in connective_patterns.items():
            for pattern in patterns:
                for match in re.finditer(r'\b' + pattern + r'\b', text):
                    connectives.append({
                        'text': pattern,
                        'type': conn_type,
                        'position': match.start()
                    })
        
        return sorted(connectives, key=lambda x: x['position'])
    
    def _analyze_references(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        """Analisa as referências e cadeias de correferência no texto."""
        pronouns = sum(1 for token in doc if token.pos_ == 'PRON')
        demonstratives = sum(1 for token in doc if token.pos_ == 'DET' and token.dep_ == 'det')
        
        # Score simplificado de coesão referencial
        score = round((pronouns + demonstratives) / len(list(doc.sents)), 2)
        
        return {
            'pronouns': pronouns,
            'demonstratives': demonstratives,
            'score': score
        }
    
    def _calculate_sentence_similarities(self, sentences: List[spacy.tokens.Span]) -> float:
        """Calcula a similaridade média entre sentenças adjacentes."""
        if len(sentences) < 2:
            return 0.0
            
        similarities = []
        for i in range(len(sentences) - 1):
            similarity = sentences[i].similarity(sentences[i + 1])
            similarities.append(similarity)
            
        return round(sum(similarities) / len(similarities), 2)

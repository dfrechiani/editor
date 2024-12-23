import streamlit as st
import openai
import json
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import time
from pathlib import Path
import base64
from io import BytesIO


import streamlit as st

# Configuração da página DEVE ser a primeira chamada Streamlit
st.set_page_config(
    page_title="Tutor de Redação ENEM",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .competencia-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .feedback-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .score-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .annotation {
        background-color: rgba(255, 220, 100, 0.2);
        border-bottom: 2px solid #ffd700;
        cursor: help;
        position: relative;
    }
    .tooltip {
        visibility: hidden;
        background-color: #333;
        color: white;
        text-align: center;
        padding: 5px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 100%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .annotation:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .exercise-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .progress-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-container {
        text-align: center;
        padding: 1rem;
    }
    .feedback-positive {
        color: #28a745;
    }
    .feedback-negative {
        color: #dc3545;
    }
    .feedback-neutral {
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Enums e Classes Base
class NivelAluno(Enum):
    INICIANTE = "iniciante"
    INTERMEDIARIO = "intermediario"
    AVANCADO = "avancado"

class CompetenciaModulo(Enum):
    NORMA_CULTA = "competencia1"
    INTERPRETACAO = "competencia2" 
    ARGUMENTACAO = "competencia3"
    COESAO = "competencia4"
    PROPOSTA = "competencia5"

@dataclass
class ProgressoCompetencia:
    nivel: float  # 0 a 1
    exercicios_feitos: int
    ultima_avaliacao: float  # 0 a 1
    pontos_fortes: List[str]
    pontos_fracos: List[str]
    data_atualizacao: datetime

@dataclass
class Redacao:
    tema: str
    texto: str
    data: datetime
    notas: Dict[CompetenciaModulo, float]
    feedback: Dict[CompetenciaModulo, List[str]]
    versao: int = 1

@dataclass
class PerfilAluno:
    nome: str
    nivel: NivelAluno
    data_inicio: datetime
    progresso_competencias: Dict[CompetenciaModulo, ProgressoCompetencia]
    historico_redacoes: List[Redacao]
    feedback_acumulado: Dict[CompetenciaModulo, List[str]]
    ultima_atividade: datetime
    total_exercicios: int = 0
    medalhas: List[str] = None

@dataclass
class ExercicioRedacao:
    tipo: str
    competencia: CompetenciaModulo
    nivel: NivelAluno
    enunciado: str
    instrucoes: List[str]
    criterios: List[str]
    exemplo_resposta: Optional[str] = None
    dicas: List[str] = None
    tempo_estimado: int = 15  # minutos

# Configurações e constantes
MAX_RETRIES = 3
API_TIMEOUT = 30.0
CACHE_TTL = 3600  # 1 hora em segundos

# Cache simples
class Cache:
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.data:
            if (datetime.now() - self.timestamps[key]).total_seconds() < CACHE_TTL:
                return self.data[key]
            else:
                del self.data[key]
                del self.timestamps[key]
        return None

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.timestamps[key] = datetime.now()

# Inicialização do cache global
cache = Cache()

# Classe base para gerenciamento de estado
class EstadoManager:
    @staticmethod
    def init_session_state():
        """Inicializa variáveis de estado da sessão"""
        if 'perfil_aluno' not in st.session_state:
            st.session_state.perfil_aluno = None
        
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = openai.OpenAI(
                api_key=st.secrets["OPENAI_API_KEY"]
            )
        
        if 'pagina_atual' not in st.session_state:
            st.session_state.pagina_atual = "inicio"
        
        if 'ultima_analise' not in st.session_state:
            st.session_state.ultima_analise = None
        
        if 'historico_exercicios' not in st.session_state:
            st.session_state.historico_exercicios = []

    @staticmethod
    def salvar_perfil():
        """Salva perfil do aluno (simulado - em produção usaria banco de dados)"""
        if st.session_state.perfil_aluno:
            st.session_state.perfil_aluno.ultima_atividade = datetime.now()

    @staticmethod
    def atualizar_progresso(competencia: CompetenciaModulo, nota: float):
        """Atualiza o progresso em uma competência específica"""
        if st.session_state.perfil_aluno:
            if competencia not in st.session_state.perfil_aluno.progresso_competencias:
                st.session_state.perfil_aluno.progresso_competencias[competencia] = \
                    ProgressoCompetencia(
                        nivel=0.0,
                        exercicios_feitos=0,
                        ultima_avaliacao=0.0,
                        pontos_fortes=[],
                        pontos_fracos=[],
                        data_atualizacao=datetime.now()
                    )
            
            progresso = st.session_state.perfil_aluno.progresso_competencias[competencia]
            progresso.ultima_avaliacao = nota
            progresso.data_atualizacao = datetime.now()
            EstadoManager.salvar_perfil()

# Classe base para análise de texto
class AnalisadorTexto:
    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.cache = Cache()

    async def analisar_texto_base(
        self, 
        texto: str, 
        sistema_prompt: str,
        prompt_template: str,
        temperatura: float = 0.7,
        retry_count: int = 0
    ) -> Dict:
        """Método base para análise de texto usando GPT"""
        if retry_count >= MAX_RETRIES:
            raise Exception("Número máximo de tentativas excedido")
            
        try:
            # Verifica cache
            cache_key = f"{hash(texto)}:{hash(prompt_template)}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

            # Faz requisição ao GPT
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": sistema_prompt},
                    {"role": "user", "content": prompt_template.format(texto=texto)}
                ],
                temperature=temperatura,
                timeout=API_TIMEOUT
            )
            
            # Processa resposta
            try:
                resultado = json.loads(response.choices[0].message.content)
                self.cache.set(cache_key, resultado)
                return resultado
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar JSON: {e}")
                if retry_count < MAX_RETRIES:
                    time.sleep(1)
                    return await self.analisar_texto_base(
                        texto, 
                        sistema_prompt,
                        prompt_template,
                        temperatura,
                        retry_count + 1
                    )
                raise
                
        except Exception as e:
            logger.error(f"Erro na análise de texto: {e}")
            if retry_count < MAX_RETRIES:
                time.sleep(1)
                return await self.analisar_texto_base(
                    texto, 
                    sistema_prompt,
                    prompt_template,
                    temperatura,
                    retry_count + 1
                )
            raise

    def destacar_texto(self, texto: str, analise: Dict) -> str:
        """Destaca elementos no texto baseado na análise"""
        texto_html = texto
        
        # Adiciona marcações HTML para elementos importantes
        if "elementos_destacados" in analise:
            for elemento in analise["elementos_destacados"]:
                inicio = elemento["posicao_inicio"]
                fim = elemento["posicao_fim"]
                tipo = elemento["tipo"]
                comentario = elemento.get("comentario", "")
                
                texto_html = (
                    texto_html[:inicio] +
                    f'<span class="annotation" data-tipo="{tipo}" title="{comentario}">' +
                    texto_html[inicio:fim] +
                    '</span>' +
                    texto_html[fim:]
                )
        
        return texto_html

    def gerar_feedback_visual(self, analise: Dict) -> str:
        """Gera feedback visual baseado na análise"""
        feedback_html = "<div class='feedback-container'>"
        
        if "pontos_positivos" in analise:
            feedback_html += "<div class='feedback-section positive'>"
            feedback_html += "<h4>✅ Pontos Positivos</h4>"
            for ponto in analise["pontos_positivos"]:
                feedback_html += f"<p class='feedback-item'>• {ponto}</p>"
            feedback_html += "</div>"
        
        if "pontos_melhoria" in analise:
            feedback_html += "<div class='feedback-section improvement'>"
            feedback_html += "<h4>💡 Pontos para Melhorar</h4>"
            for ponto in analise["pontos_melhoria"]:
                feedback_html += f"<p class='feedback-item'>• {ponto}</p>"
            feedback_html += "</div>"
        
        feedback_html += "</div>"
        return feedback_html

class ModuloBase:
    """Classe base para todos os módulos de competência"""
    
    def __init__(self, client: openai.OpenAI, competencia: CompetenciaModulo):
        self.client = client
        self.competencia = competencia
        self.cache = Cache()
        self.analisador = AnalisadorTexto(client)

    async def _fazer_requisicao_gpt(
        self, 
        prompt: str, 
        sistema_prompt: str,
        temperatura: float = 0.7,
        retry_count: int = 0
    ) -> Dict:
        """Método base para fazer requisições ao GPT-4"""
        return await self.analisador.analisar_texto_base(
            prompt,
            sistema_prompt,
            prompt,
            temperatura,
            retry_count
        )

    def calcular_nota(self, analise: Dict) -> float:
        """Calcula nota de 0-200 baseada na análise"""
        try:
            if "score_geral" in analise:
                return float(analise["score_geral"])
            return sum(
                criterio["score"] 
                for criterio in analise.get("criterios", [])
            ) / len(analise.get("criterios", [1])) * 200
        except Exception as e:
            logger.error(f"Erro ao calcular nota: {e}")
            return 0.0

class ModuloNormaCulta(ModuloBase):
    """Módulo específico para Competência 1 - Domínio da Norma Culta"""
    
    def __init__(self, client: openai.OpenAI):
        super().__init__(client, CompetenciaModulo.NORMA_CULTA)
        self.sistema_prompt = """Você é um tutor especializado na primeira competência do ENEM:
        domínio da norma culta da língua escrita. Você deve:
        
        1. Analisar aspectos formais da escrita:
           - Ortografia
           - Acentuação
           - Pontuação
           - Concordância
           - Regência
           - Colocação pronominal
        
        2. Verificar adequação vocabular:
           - Registro formal
           - Precisão lexical
           - Variação vocabular
        
        3. Identificar problemas de construção:
           - Paralelismo
           - Ambiguidade
           - Redundância
           - Repetições
        
        Forneça feedback construtivo e específico, sempre explicando o porquê das correções
        e sugerindo formas de melhorar."""

    async def analisar_texto(self, texto: str) -> Dict:
        """Analisa o texto quanto à norma culta"""
        prompt = f"""Analise detalhadamente o seguinte texto quanto ao domínio da norma culta:

        TEXTO: {texto}

        Retorne um JSON com:
        {{
            "erros_gramaticais": [
                {{
                    "tipo": "tipo do erro",
                    "trecho": "trecho com erro",
                    "correcao": "sugestão de correção",
                    "explicacao": "explicação didática",
                    "posicao": [inicio, fim]
                }}
            ],
            "adequacao_vocabular": {{
                "nivel_formalidade": 1-5,
                "problemas_identificados": [],
                "sugestoes_melhoria": [],
                "termos_inadequados": [
                    {{
                        "termo": "termo encontrado",
                        "sugestao": "termo mais adequado",
                        "posicao": [inicio, fim]
                    }}
                ]
            }},
            "construcao_frases": {{
                "problemas": [
                    {{
                        "tipo": "tipo do problema",
                        "trecho": "trecho problemático",
                        "sugestao": "como melhorar",
                        "posicao": [inicio, fim]
                    }}
                ],
                "sugestoes": []
            }},
            "pontuacao": {{
                "erros": [
                    {{
                        "tipo": "tipo do erro",
                        "trecho": "trecho com erro",
                        "correcao": "correção sugerida",
                        "posicao": [inicio, fim]
                    }}
                ],
                "sugestoes": []
            }},
            "score_geral": 0-200,
            "feedback_geral": "feedback construtivo",
            "elementos_destacados": [
                {{
                    "tipo": "tipo do elemento",
                    "texto": "texto destacado",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "comentário sobre o elemento"
                }}
            ],
            "proximos_passos": []
        }}"""

        resultado = await self._fazer_requisicao_gpt(
            texto,
            self.sistema_prompt
        )
        
        # Adiciona elementos destacados para visualização
        resultado["elementos_destacados"] = (
            self._gerar_elementos_destacados(resultado)
        )
        
        return resultado

    def _gerar_elementos_destacados(self, analise: Dict) -> List[Dict]:
        """Gera lista de elementos para destacar no texto"""
        elementos = []
        
        # Adiciona erros gramaticais
        for erro in analise.get("erros_gramaticais", []):
            elementos.append({
                "tipo": "erro_gramatical",
                "texto": erro["trecho"],
                "posicao_inicio": erro["posicao"][0],
                "posicao_fim": erro["posicao"][1],
                "comentario": erro["explicacao"]
            })
        
        # Adiciona problemas de vocabulário
        for termo in analise.get("adequacao_vocabular", {}).get("termos_inadequados", []):
            elementos.append({
                "tipo": "vocabulario_inadequado",
                "texto": termo["termo"],
                "posicao_inicio": termo["posicao"][0],
                "posicao_fim": termo["posicao"][1],
                "comentario": f"Sugestão: {termo['sugestao']}"
            })
        
        # Adiciona problemas de construção
        for problema in analise.get("construcao_frases", {}).get("problemas", []):
            elementos.append({
                "tipo": "problema_construcao",
                "texto": problema["trecho"],
                "posicao_inicio": problema["posicao"][0],
                "posicao_fim": problema["posicao"][1],
                "comentario": problema["sugestao"]
            })
        
        return elementos

    async def gerar_exercicios(
        self, 
        nivel: NivelAluno,
        areas_foco: List[str]
    ) -> List[ExercicioRedacao]:
        """Gera exercícios personalizados de norma culta"""
        prompt = f"""Crie exercícios de norma culta para nível {nivel.value} 
        focando nas áreas: {', '.join(areas_foco)}

        Retorne um JSON com exercícios seguindo exatamente este formato:
        {{
            "exercicios": [
                {{
                    "tipo": "tipo do exercício",
                    "nivel": "{nivel.value}",
                    "enunciado": "enunciado completo",
                    "instrucoes": ["instrução 1", "instrução 2"],
                    "criterios": ["critério 1", "critério 2"],
                    "exemplo_resposta": "exemplo de resposta esperada",
                    "dicas": ["dica 1", "dica 2"],
                    "tempo_estimado": tempo_em_minutos
                }}
            ]
        }}"""

        resultado = await self._fazer_requisicao_gpt(
            prompt,
            self.sistema_prompt
        )
        
        return [
            ExercicioRedacao(**ex) 
            for ex in resultado.get("exercicios", [])
        ]


class ModuloInterpretacao(ModuloBase):
    """Módulo específico para Competência 2 - Compreensão da proposta"""
    
    def __init__(self, client: openai.OpenAI):
        super().__init__(client, CompetenciaModulo.INTERPRETACAO)
        self.sistema_prompt = """Você é um tutor especializado na segunda competência do ENEM:
        compreensão da proposta e desenvolvimento do tema. Você deve:
        
        1. Auxiliar na análise dos textos motivadores:
           - Identificação de ideias principais
           - Relações entre os textos
           - Contextualização do tema
           - Identificação de dados relevantes
        
        2. Orientar na compreensão do tema:
           - Palavras-chave
           - Delimitação do recorte temático
           - Aspectos centrais e periféricos
           - Abordagens possíveis
        
        3. Verificar a pertinência temática:
           - Aderência ao tema
           - Tangenciamento
           - Fuga do tema
           - Aprofundamento adequado
        
        Forneça orientações construtivas que levem o aluno a desenvolver autonomia
        na interpretação de propostas de redação."""

    async def analisar_compreensao(
        self, 
        tema: str, 
        textos_motivadores: List[str], 
        texto_aluno: str
    ) -> Dict:
        """Analisa a compreensão do tema e textos motivadores"""
        prompt = f"""Analise a compreensão do tema e dos textos motivadores:

        TEMA: {tema}
        
        TEXTOS MOTIVADORES:
        {json.dumps(textos_motivadores)}
        
        TEXTO DO ALUNO:
        {texto_aluno}

        Retorne um JSON com:
        {{
            "analise_tema": {{
                "palavras_chave": [],
                "recorte_identificado": "descrição do recorte",
                "abordagem_aluno": "descrição da abordagem",
                "pertinencia": 0-200,
                "problemas": [],
                "elementos_destacados": [
                    {{
                        "tipo": "palavra_chave",
                        "texto": "termo encontrado",
                        "posicao_inicio": inicio,
                        "posicao_fim": fim,
                        "comentario": "relevância do termo"
                    }}
                ]
            }},
            "uso_motivadores": {{
                "referencias_explicitas": [
                    {{
                        "texto": "referência encontrada",
                        "texto_original": "trecho do motivador",
                        "tipo_uso": "tipo de referência",
                        "posicao": [inicio, fim]
                    }}
                ],
                "referencias_implicitas": [],
                "integracao_argumentos": 0-200,
                "sugestoes_uso": []
            }},
            "desenvolvimento": {{
                "aspectos_contemplados": [],
                "aspectos_ignorados": [],
                "nivel_aprofundamento": 0-200,
                "sugestoes_desenvolvimento": []
            }},
            "score_geral": 0-200,
            "feedback_detalhado": "feedback construtivo",
            "proximos_passos": [],
            "elementos_destacados": [
                {{
                    "tipo": "tipo do elemento",
                    "texto": "texto destacado",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "comentário sobre o elemento"
                }}
            ]
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return self._processar_resultado_analise(resultado, texto_aluno)

    def _processar_resultado_analise(self, resultado: Dict, texto_original: str) -> Dict:
        """Processa o resultado da análise adicionando elementos visuais"""
        elementos_destacados = []
        
        # Adiciona palavras-chave
        if "analise_tema" in resultado:
            elementos_destacados.extend(
                resultado["analise_tema"].get("elementos_destacados", [])
            )
        
        # Adiciona referências aos textos motivadores
        if "uso_motivadores" in resultado:
            for ref in resultado["uso_motivadores"].get("referencias_explicitas", []):
                elementos_destacados.append({
                    "tipo": "referencia_motivador",
                    "texto": ref["texto"],
                    "posicao_inicio": ref["posicao"][0],
                    "posicao_fim": ref["posicao"][1],
                    "comentario": f"Referência ao texto motivador: {ref['texto_original']}"
                })
        
        resultado["elementos_destacados"] = elementos_destacados
        return resultado

    async def gerar_exercicios_interpretacao(
        self,
        nivel: NivelAluno,
        foco: List[str]
    ) -> List[ExercicioRedacao]:
        """Gera exercícios de interpretação de propostas"""
        prompt = f"""Crie exercícios de interpretação para nível {nivel.value} 
        focando em: {', '.join(foco)}

        Retorne um JSON com:
        {{
            "exercicios": [
                {{
                    "tipo": "interpretacao",
                    "nivel": "{nivel.value}",
                    "enunciado": "enunciado completo",
                    "textos_motivadores": [],
                    "instrucoes": ["instrução 1", "instrução 2"],
                    "criterios": ["critério 1", "critério 2"],
                    "exemplo_resposta": "exemplo de resposta esperada",
                    "dicas": ["dica 1", "dica 2"],
                    "tempo_estimado": tempo_em_minutos
                }}
            ]
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return [
            ExercicioRedacao(**ex) 
            for ex in resultado.get("exercicios", [])
        ]

    async def analisar_textos_motivadores(
        self,
        textos: List[str],
        tema: str
    ) -> Dict:
        """Auxilia na análise dos textos motivadores"""
        prompt = f"""Analise os textos motivadores em relação ao tema:

        TEMA: {tema}
        
        TEXTOS:
        {json.dumps(textos)}

        Retorne um JSON com:
        {{
            "analise_individual": [
                {{
                    "texto": "texto analisado",
                    "ideias_principais": [],
                    "dados_relevantes": [],
                    "relacao_tema": "explicação",
                    "possiveis_usos": []
                }}
            ],
            "relacoes_entre_textos": [],
            "aspectos_complementares": [],
            "sugestoes_abordagem": [],
            "armadilhas_evitar": [],
            "elementos_destacados": [
                {{
                    "tipo": "ideia_principal",
                    "texto": "trecho relevante",
                    "texto_original": "número do texto motivador",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "relevância do trecho"
                }}
            ]
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

    async def gerar_roteiro_interpretacao(self, tema: str, nivel: NivelAluno) -> Dict:
        """Gera um roteiro para interpretação do tema"""
        prompt = f"""Crie um roteiro de interpretação para o tema:
        
        TEMA: {tema}
        NÍVEL: {nivel.value}

        Retorne um JSON com:
        {{
            "etapas_analise": [
                {{
                    "ordem": número_da_etapa,
                    "descricao": "o que fazer",
                    "objetivo": "por que fazer",
                    "dicas": ["dica 1", "dica 2"]
                }}
            ],
            "perguntas_guia": [
                {{
                    "pergunta": "pergunta orientadora",
                    "objetivo": "objetivo da pergunta",
                    "dicas_reflexao": []
                }}
            ],
            "armadilhas_comuns": [],
            "estrategias_foco": []
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

class ModuloArgumentacao(ModuloBase):
    """Módulo específico para Competência 3 - Argumentação"""
    
    def __init__(self, client: openai.OpenAI):
        super().__init__(client, CompetenciaModulo.ARGUMENTACAO)
        self.sistema_prompt = """Você é um tutor especializado na terceira competência do ENEM:
        argumentação e defesa de ponto de vista. Você deve analisar:
        
        1. ESTRUTURA ARGUMENTATIVA
           - Projeto de texto (argumentos principais e secundários)
           - Seleção estratégica de informações
           - Hierarquização de ideias
           - Desenvolvimento progressivo
        
        2. TIPOS DE ARGUMENTOS
           - Causa e consequência
           - Exemplificação
           - Comparação
           - Dados estatísticos
           - Argumento de autoridade
           - Contraposição
        
        3. REPERTÓRIO SOCIOCULTURAL
           - Filosófico
           - Histórico
           - Literário
           - Sociológico
           - Científico
           - Atualidades
        
        4. QUALIDADE ARGUMENTATIVA
           - Pertinência
           - Profundidade
           - Produtividade
           - Circularidade vs. Progressão
           - Consistência
        
        Forneça feedback específico e orientações práticas para desenvolvimento
        de argumentação sólida dentro dos critérios do ENEM."""

    async def analisar_argumentacao(self, texto: str, tema: str) -> Dict:
        """Analisa detalhadamente a argumentação do texto"""
        prompt = f"""Analise a argumentação no texto:

        TEMA: {tema}
        TEXTO: {texto}

        Retorne um JSON com:
        {{
            "estrutura_argumentativa": {{
                "tese": {{
                    "texto": "tese identificada",
                    "clareza": 0-200,
                    "posicao": [inicio, fim],
                    "tipo_desenvolvimento": "análise"
                }},
                "argumentos": [
                    {{
                        "tipo": "tipo do argumento",
                        "texto": "texto do argumento",
                        "posicao": [inicio, fim],
                        "forca": 0-200,
                        "desenvolvimento": "análise",
                        "problemas": [],
                        "sugestoes": []
                    }}
                ],
                "hierarquia_ideias": "análise da hierarquização"
            }},
            "repertorio_sociocultural": {{
                "referencias": [
                    {{
                        "tipo": "área do conhecimento",
                        "texto": "referência utilizada",
                        "posicao": [inicio, fim],
                        "pertinencia": 0-200,
                        "desenvolvimento": "análise do uso"
                    }}
                ],
                "areas_presentes": [],
                "areas_ausentes": [],
                "qualidade_uso": 0-200,
                "sugestoes_ampliacao": []
            }},
            "progressao_argumentativa": {{
                "encadeamento_logico": 0-200,
                "aprofundamento": 0-200,
                "problemas_identificados": [],
                "pontos_fortes": []
            }},
            "avaliacao_criterios": {{
                "pertinencia": 0-200,
                "produtividade": 0-200,
                "circularidade": "análise",
                "consistencia": 0-200
            }},
            "score_geral": 0-200,
            "feedback_detalhado": "feedback construtivo",
            "elementos_destacados": [
                {{
                    "tipo": "tipo do elemento",
                    "texto": "texto destacado",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "comentário sobre o elemento"
                }}
            ],
            "sugestoes_melhoria": []
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return self._processar_resultado_argumentacao(resultado)

    def _processar_resultado_argumentacao(self, resultado: Dict) -> Dict:
        """Processa o resultado da análise argumentativa"""
        elementos_destacados = []
        
        # Destaca tese
        if "estrutura_argumentativa" in resultado:
            tese = resultado["estrutura_argumentativa"].get("tese")
            if tese:
                elementos_destacados.append({
                    "tipo": "tese",
                    "texto": tese["texto"],
                    "posicao_inicio": tese["posicao"][0],
                    "posicao_fim": tese["posicao"][1],
                    "comentario": f"Tese - Clareza: {tese['clareza']}/200"
                })
        
        # Destaca argumentos
        for arg in resultado["estrutura_argumentativa"].get("argumentos", []):
            elementos_destacados.append({
                "tipo": f"argumento_{arg['tipo']}",
                "texto": arg["texto"],
                "posicao_inicio": arg["posicao"][0],
                "posicao_fim": arg["posicao"][1],
                "comentario": f"{arg['tipo'].title()} - Força: {arg['forca']}/200"
            })
        
        # Destaca repertório
        for ref in resultado["repertorio_sociocultural"].get("referencias", []):
            elementos_destacados.append({
                "tipo": f"repertorio_{ref['tipo']}",
                "texto": ref["texto"],
                "posicao_inicio": ref["posicao"][0],
                "posicao_fim": ref["posicao"][1],
                "comentario": f"Repertório {ref['tipo']} - Pertinência: {ref['pertinencia']}/200"
            })
        
        resultado["elementos_destacados"] = elementos_destacados
        return resultado

    async def sugerir_repertorio(self, tema: str, nivel: NivelAluno) -> Dict:
        """Sugere repertório sociocultural relevante"""
        prompt = f"""Sugira repertório sociocultural para o tema:
        
        TEMA: {tema}
        NÍVEL: {nivel.value}

        Retorne um JSON com:
        {{
            "areas_conhecimento": [
                {{
                    "area": "área do conhecimento",
                    "exemplos": [
                        {{
                            "conteudo": "exemplo específico",
                            "aplicacao": "como aplicar",
                            "fonte": "referência",
                            "nivel_complexidade": 1-5
                        }}
                    ]
                }}
            ],
            "argumentos_possiveis": [
                {{
                    "tipo": "tipo de argumento",
                    "desenvolvimento": "como desenvolver",
                    "repertorio_sugerido": [],
                    "exemplo_uso": "exemplo de aplicação"
                }}
            ],
            "material_aprofundamento": {{
                "artigos": [],
                "videos": [],
                "livros": []
            }},
            "dicas_uso": []
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

    async def gerar_exercicios_argumentacao(
        self,
        nivel: NivelAluno,
        foco: List[str]
    ) -> List[ExercicioRedacao]:
        """Gera exercícios específicos de argumentação"""
        prompt = f"""Crie exercícios de argumentação para nível {nivel.value}
        focando em: {', '.join(foco)}

        Retorne um JSON com:
        {{
            "exercicios": [
                {{
                    "tipo": "argumentacao",
                    "nivel": "{nivel.value}",
                    "enunciado": "enunciado completo",
                    "contexto": "contextualização",
                    "instrucoes": ["instrução 1", "instrução 2"],
                    "criterios": ["critério 1", "critério 2"],
                    "exemplo_resposta": "exemplo de resposta esperada",
                    "dicas": ["dica 1", "dica 2"],
                    "tempo_estimado": tempo_em_minutos
                }}
            ],
            "material_apoio": {{
                "tecnicas": [],
                "exemplos": [],
                "repertorio": []
            }}
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return [
            ExercicioRedacao(**ex) 
            for ex in resultado.get("exercicios", [])
        ]

    async def analisar_progressao_argumentativa(self, texto: str) -> Dict:
        """Analisa a progressão e encadeamento dos argumentos"""
        prompt = f"""Analise a progressão argumentativa do texto:

        TEXTO: {texto}

        Retorne um JSON com:
        {{
            "encadeamento": {{
                "sequencia_logica": 0-200,
                "transicoes": "análise das transições",
                "problemas_identificados": []
            }},
            "desenvolvimento": {{
                "aprofundamento": 0-200,
                "circularidade": "análise",
                "pontos_criticos": []
            }},
            "elementos_coesivos": [
                {{
                    "texto": "elemento coesivo",
                    "funcao": "função no texto",
                    "posicao": [inicio, fim],
                    "eficacia": 0-200
                }}
            ],
            "mapa_argumentativo": {{
                "estrutura": "descrição da estrutura",
                "fluxo": "análise do fluxo",
                "sugestoes_melhoria": []
            }}
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

class ModuloCoesao(ModuloBase):
    """Módulo específico para Competência 4 - Coesão Textual"""
    
    def __init__(self, client: openai.OpenAI):
        super().__init__(client, CompetenciaModulo.COESAO)
        self.sistema_prompt = """Você é um tutor especializado na quarta competência do ENEM:
        mecanismos linguísticos e coesão textual. Você deve analisar:
        
        1. RECURSOS COESIVOS
           - Conectivos e operadores argumentativos
           - Pronomes e elementos referenciais
           - Elipse e substituição
           - Repetição e paralelismo
           - Articuladores de coesão
        
        2. ARTICULAÇÃO TEXTUAL
           - Encadeamento de parágrafos
           - Relações lógico-semânticas
           - Progressão temática
           - Coerência argumentativa
           - Fluidez do texto
        
        3. QUALIDADE DA COESÃO
           - Precisão dos conectivos
           - Clareza das referências
           - Adequação das transições
           - Manutenção temática
           - Progressão textual
        
        Forneça feedback específico e orientações práticas para desenvolvimento
        de texto coeso e bem articulado."""

    async def analisar_coesao(self, texto: str) -> Dict:
        """Analisa detalhadamente os mecanismos de coesão"""
        prompt = f"""Analise os mecanismos de coesão no texto:

        TEXTO: {texto}

        Retorne um JSON com:
        {{
            "recursos_coesivos": {{
                "conectivos": [
                    {{
                        "texto": "conectivo usado",
                        "tipo": "tipo de conexão",
                        "posicao": [inicio, fim],
                        "funcao": "função no texto",
                        "eficacia": 0-200,
                        "sugestoes": []
                    }}
                ],
                "elementos_referenciais": [
                    {{
                        "texto": "elemento referencial",
                        "referente": "a que se refere",
                        "posicao": [inicio, fim],
                        "clareza": 0-200,
                        "problemas": []
                    }}
                ],
                "repeticoes": [
                    {{
                        "texto": "termo repetido",
                        "ocorrencias": [posicoes],
                        "tipo": "intencional/problemática",
                        "sugestoes": []
                    }}
                ]
            }},
            "articulacao_textual": {{
                "entre_paragrafos": {{
                    "transicoes": [
                        {{
                            "posicao": [inicio, fim],
                            "tipo": "tipo de transição",
                            "qualidade": 0-200,
                            "sugestoes": []
                        }}
                    ],
                    "problemas_identificados": []
                }},
                "dentro_paragrafos": {{
                    "qualidade": 0-200,
                    "problemas": [],
                    "sugestoes": []
                }}
            }},
            "progressao_tematica": {{
                "manutencao_tema": 0-200,
                "desenvolvimento": "análise do desenvolvimento",
                "quebras": [
                    {{
                        "posicao": [inicio, fim],
                        "problema": "descrição do problema",
                        "sugestao": "como corrigir"
                    }}
                ]
            }},
            "score_geral": 0-200,
            "feedback_detalhado": "feedback construtivo",
            "elementos_destacados": [
                {{
                    "tipo": "tipo do elemento",
                    "texto": "texto destacado",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "comentário sobre o elemento"
                }}
            ],
            "sugestoes_melhoria": []
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return self._processar_resultado_coesao(resultado)

    def _processar_resultado_coesao(self, resultado: Dict) -> Dict:
        """Processa o resultado da análise de coesão"""
        elementos_destacados = []
        
        # Destaca conectivos
        for conectivo in resultado["recursos_coesivos"].get("conectivos", []):
            elementos_destacados.append({
                "tipo": "conectivo",
                "texto": conectivo["texto"],
                "posicao_inicio": conectivo["posicao"][0],
                "posicao_fim": conectivo["posicao"][1],
                "comentario": f"{conectivo['tipo'].title()} - Eficácia: {conectivo['eficacia']}/200"
            })
        
        # Destaca elementos referenciais
        for ref in resultado["recursos_coesivos"].get("elementos_referenciais", []):
            elementos_destacados.append({
                "tipo": "referencial",
                "texto": ref["texto"],
                "posicao_inicio": ref["posicao"][0],
                "posicao_fim": ref["posicao"][1],
                "comentario": f"Referente a: {ref['referente']} - Clareza: {ref['clareza']}/200"
            })
        
        # Destaca repetições problemáticas
        for rep in resultado["recursos_coesivos"].get("repeticoes", []):
            if rep["tipo"] == "problemática":
                for pos in rep["ocorrencias"]:
                    elementos_destacados.append({
                        "tipo": "repeticao",
                        "texto": rep["texto"],
                        "posicao_inicio": pos[0],
                        "posicao_fim": pos[1],
                        "comentario": "Repetição que pode ser evitada"
                    })
        
        # Destaca quebras de progressão
        for quebra in resultado["progressao_tematica"].get("quebras", []):
            elementos_destacados.append({
                "tipo": "quebra_progressao",
                "texto": quebra["texto"],
                "posicao_inicio": quebra["posicao"][0],
                "posicao_fim": quebra["posicao"][1],
                "comentario": quebra["problema"]
            })
        
        resultado["elementos_destacados"] = elementos_destacados
        return resultado

    async def gerar_exercicios_coesao(
        self,
        nivel: NivelAluno,
        foco: List[str]
    ) -> List[ExercicioRedacao]:
        """Gera exercícios específicos de coesão textual"""
        prompt = f"""Crie exercícios de coesão textual para nível {nivel.value}
        focando em: {', '.join(foco)}

        Retorne um JSON com:
        {{
            "exercicios": [
                {{
                    "tipo": "coesao",
                    "nivel": "{nivel.value}",
                    "enunciado": "enunciado completo",
                    "texto_base": "texto para exercício",
                    "instrucoes": ["instrução 1", "instrução 2"],
                    "criterios": ["critério 1", "critério 2"],
                    "exemplo_resposta": "exemplo de resposta esperada",
                    "dicas": ["dica 1", "dica 2"],
                    "tempo_estimado": tempo_em_minutos
                }}
            ],
            "material_apoio": {{
                "conectivos_essenciais": [],
                "estruturas_modelo": [],
                "exemplos_praticos": []
            }}
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return [
            ExercicioRedacao(**ex) 
            for ex in resultado.get("exercicios", [])
        ]

    async def sugerir_melhorias_coesao(
        self,
        texto: str,
        problemas_identificados: List[str]
    ) -> Dict:
        """Sugere melhorias específicas para problemas de coesão"""
        prompt = f"""Sugira melhorias de coesão para o texto, 
        considerando os problemas identificados: {', '.join(problemas_identificados)}

        TEXTO: {texto}

        Retorne um JSON com:
        {{
            "sugestoes": [
                {{
                    "problema": "descrição do problema",
                    "trecho_original": "texto original",
                    "sugestao": "como melhorar",
                    "explicacao": "por que melhorar assim",
                    "exemplos": []
                }}
            ],
            "exercicios_pratica": [
                {{
                    "foco": "aspecto a praticar",
                    "instrucoes": "como praticar",
                    "exemplo": "exemplo de prática"
                }}
            ],
            "recursos_recomendados": []
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

class ModuloProposta(ModuloBase):
    """Módulo específico para Competência 5 - Proposta de Intervenção"""
    
    def __init__(self, client: openai.OpenAI):
        super().__init__(client, CompetenciaModulo.PROPOSTA)
        self.sistema_prompt = """Você é um tutor especializado na quinta competência do ENEM:
        elaboração de proposta de intervenção. Você deve analisar:
        
        1. ELEMENTOS ESSENCIAIS
           - Agente (quem realizará a ação)
           - Ação (o que será feito)
           - Modo/Meio (como será realizado)
           - Efeito (resultado esperado)
           - Detalhamento (especificações)
        
        2. CRITÉRIOS DE AVALIAÇÃO
           - Pertinência ao tema
           - Detalhamento das ações
           - Articulação com argumentos
           - Respeito aos direitos humanos
           - Exequibilidade
        
        3. QUALIDADE DA PROPOSTA
           - Viabilidade prática
           - Abrangência
           - Inovação
           - Especificidade
           - Impacto potencial
        
        Forneça feedback específico e orientações práticas para desenvolvimento
        de propostas de intervenção efetivas e bem detalhadas."""

    async def analisar_proposta(self, texto: str, tema: str) -> Dict:
        """Analisa detalhadamente a proposta de intervenção"""
        prompt = f"""Analise a proposta de intervenção:

        TEMA: {tema}
        TEXTO: {texto}

        Retorne um JSON com:
        {{
            "elementos_proposta": {{
                "agentes": [
                    {{
                        "texto": "agente identificado",
                        "posicao": [inicio, fim],
                        "nivel_detalhamento": 0-200,
                        "sugestoes": []
                    }}
                ],
                "acoes": [
                    {{
                        "texto": "ação proposta",
                        "posicao": [inicio, fim],
                        "agente_relacionado": "agente responsável",
                        "viabilidade": 0-200,
                        "detalhamento": "análise do detalhamento"
                    }}
                ],
                "modos": [
                    {{
                        "texto": "modo de execução",
                        "posicao": [inicio, fim],
                        "acao_relacionada": "ação relacionada",
                        "praticidade": 0-200,
                        "recursos_necessarios": []
                    }}
                ],
                "efeitos": [
                    {{
                        "texto": "efeito esperado",
                        "posicao": [inicio, fim],
                        "plausibilidade": 0-200,
                        "alcance": "análise do alcance"
                    }}
                ]
            }},
            "avaliacao_criterios": {{
                "pertinencia_tema": 0-200,
                "nivel_detalhamento": 0-200,
                "articulacao_argumentos": 0-200,
                "respeito_dh": 0-200,
                "exequibilidade": 0-200,
                "problemas_identificados": []
            }},
            "analise_qualidade": {{
                "viabilidade": {{
                    "score": 0-200,
                    "aspectos_positivos": [],
                    "aspectos_negativos": [],
                    "sugestoes": []
                }},
                "abrangencia": {{
                    "score": 0-200,
                    "alcance": "análise do alcance",
                    "limitacoes": []
                }},
                "inovacao": {{
                    "score": 0-200,
                    "aspectos_inovadores": [],
                    "sugestoes_ampliacao": []
                }}
            }},
            "score_geral": 0-200,
            "feedback_detalhado": "feedback construtivo",
            "elementos_destacados": [
                {{
                    "tipo": "tipo do elemento",
                    "texto": "texto destacado",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "comentário sobre o elemento"
                }}
            ],
            "sugestoes_melhoria": []
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return self._processar_resultado_proposta(resultado)

    def _processar_resultado_proposta(self, resultado: Dict) -> Dict:
        """Processa o resultado da análise da proposta"""
        elementos_destacados = []
        
        # Destaca agentes
        for agente in resultado["elementos_proposta"].get("agentes", []):
            elementos_destacados.append({
                "tipo": "agente",
                "texto": agente["texto"],
                "posicao_inicio": agente["posicao"][0],
                "posicao_fim": agente["posicao"][1],
                "comentario": f"Agente - Detalhamento: {agente['nivel_detalhamento']}/200"
            })
        
        # Destaca ações
        for acao in resultado["elementos_proposta"].get("acoes", []):
            elementos_destacados.append({
                "tipo": "acao",
                "texto": acao["texto"],
                "posicao_inicio": acao["posicao"][0],
                "posicao_fim": acao["posicao"][1],
                "comentario": f"Ação - Viabilidade: {acao['viabilidade']}/200"
            })
        
        # Destaca modos
        for modo in resultado["elementos_proposta"].get("modos", []):
            elementos_destacados.append({
                "tipo": "modo",
                "texto": modo["texto"],
                "posicao_inicio": modo["posicao"][0],
                "posicao_fim": modo["posicao"][1],
                "comentario": f"Modo - Praticidade: {modo['praticidade']}/200"
            })
        
        # Destaca efeitos
        for efeito in resultado["elementos_proposta"].get("efeitos", []):
            elementos_destacados.append({
                "tipo": "efeito",
                "texto": efeito["texto"],
                "posicao_inicio": efeito["posicao"][0],
                "posicao_fim": efeito["posicao"][1],
                "comentario": f"Efeito - Plausibilidade: {efeito['plausibilidade']}/200"
            })
        
        resultado["elementos_destacados"] = elementos_destacados
        return resultado

    async def gerar_exercicios_proposta(
        self,
        nivel: NivelAluno,
        foco: List[str]
    ) -> List[ExercicioRedacao]:
        """Gera exercícios específicos de elaboração de proposta"""
        prompt = f"""Crie exercícios de elaboração de proposta para nível {nivel.value}
        focando em: {', '.join(foco)}

        Retorne um JSON com:
        {{
            "exercicios": [
                {{
                    "tipo": "proposta",
                    "nivel": "{nivel.value}",
                    "enunciado": "enunciado completo",
                    "contexto": "contextualização do problema",
                    "instrucoes": ["instrução 1", "instrução 2"],
                    "criterios": ["critério 1", "critério 2"],
                    "exemplo_resposta": "exemplo de proposta",
                    "dicas": ["dica 1", "dica 2"],
                    "tempo_estimado": tempo_em_minutos
                }}
            ],
            "material_apoio": {{
                "estruturas_modelo": [],
                "exemplos_praticos": [],
                "dicas_detalhamento": []
            }}
        }}"""

        resultado = await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)
        return [
            ExercicioRedacao(**ex) 
            for ex in resultado.get("exercicios", [])
        ]

    async def verificar_direitos_humanos(self, proposta: str) -> Dict:
        """Verifica o respeito aos direitos humanos na proposta"""
        prompt = f"""Analise o respeito aos direitos humanos na proposta:

        PROPOSTA: {proposta}

        Retorne um JSON com:
        {{
            "analise_dh": {{
                "conformidade": 0-200,
                "problemas_identificados": [
                    {{
                        "texto": "trecho problemático",
                        "posicao": [inicio, fim],
                        "problema": "descrição do problema",
                        "sugestao": "como corrigir"
                    }}
                ],
                "aspectos_positivos": []
            }},
            "grupos_afetados": [
                {{
                    "grupo": "grupo social",
                    "impacto": "análise do impacto",
                    "consideracoes": [],
                    "recomendacoes": []
                }}
            ],
            "elementos_destacados": [
                {{
                    "tipo": "violacao_dh",
                    "texto": "texto destacado",
                    "posicao_inicio": inicio,
                    "posicao_fim": fim,
                    "comentario": "comentário sobre a violação"
                }}
            ],
            "sugestoes_ajuste": []
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

    async def sugerir_detalhamento(self, proposta: str, nivel: NivelAluno) -> Dict:
        """Sugere formas de detalhar melhor a proposta"""
        prompt = f"""Sugira formas de detalhar a proposta para nível {nivel.value}:

        PROPOSTA: {proposta}

        Retorne um JSON com:
        {{
            "aspectos_detalhamento": [
                {{
                    "elemento": "elemento a detalhar",
                    "situacao_atual": "análise atual",
                    "sugestoes": [],
                    "exemplos": []
                }}
            ],
            "modelos_detalhamento": [
                {{
                    "tipo": "tipo de detalhamento",
                    "estrutura": "como estruturar",
                    "exemplo": "exemplo prático"
                }}
            ],
            "exercicios_pratica": [],
            "dicas_especificas": []
        }}"""

        return await self._fazer_requisicao_gpt(prompt, self.sistema_prompt)

# app.py - Arquivo Principal

# Configuração inicial do Streamlit
st.set_page_config(
    page_title="Tutor de Redação ENEM",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização do estado
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = openai.OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )
    if 'perfil_aluno' not in st.session_state:
        st.session_state.perfil_aluno = None
    if 'ultima_analise' not in st.session_state:
        st.session_state.ultima_analise = None

def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📝 Tutor ENEM")
        
        if st.session_state.perfil_aluno:
            st.markdown(f"""
            ### 👋 Olá, {st.session_state.perfil_aluno.nome}!
            Nível: {st.session_state.perfil_aluno.nivel.value.title()}
            """)
            
            menu_items = [
                "Dashboard",
                "Nova Redação",
                "Competência 1 - Norma Culta",
                "Competência 2 - Compreensão",
                "Competência 3 - Argumentação",
                "Competência 4 - Coesão",
                "Competência 5 - Proposta",
                "Exercícios",
                "Meu Progresso"
            ]
        else:
            menu_items = ["Início"]
        
        menu_choice = st.radio("Menu", menu_items)
        st.session_state.page = menu_choice.lower().replace(" ", "_")
        
        if st.session_state.perfil_aluno:
            if st.button("Sair"):
                st.session_state.perfil_aluno = None
                st.experimental_rerun()
    
    # Conteúdo principal
    if st.session_state.page == 'inicio':
        show_inicio_page()
    elif st.session_state.page == 'dashboard':
        show_dashboard_page()
    elif st.session_state.page == 'nova_redação':
        show_nova_redacao_page()
    elif st.session_state.page.startswith('competência'):
        show_competencia_page(st.session_state.page)
    elif st.session_state.page == 'exercícios':
        show_exercicios_page()
    elif st.session_state.page == 'meu_progresso':
        show_progresso_page()

def show_inicio_page():
    st.title("🎓 Bem-vindo ao Tutor de Redação ENEM")
    
    st.markdown("""
    ### Comece sua jornada de preparação!
    
    Este tutor inteligente vai te ajudar a desenvolver todas as competências 
    necessárias para uma excelente redação no ENEM.
    """)
    
    with st.form("cadastro_inicial"):
        nome = st.text_input("Como podemos te chamar?")
        nivel = st.selectbox(
            "Qual seu nível atual em redação?",
            ["Iniciante", "Intermediário", "Avançado"]
        )
        
        texto_diagnostico = st.text_area(
            "Para começar, escreva um pequeno parágrafo sobre qualquer tema atual:",
            height=150,
            help="Isso nos ajudará a personalizar seu aprendizado"
        )
        
        submitted = st.form_submit_button("Começar")
        
        if submitted and nome and texto_diagnostico:
            with st.spinner("Analisando seu perfil..."):
                try:
                    nivel_enum = NivelAluno[nivel.upper()]
                    st.session_state.perfil_aluno = PerfilAluno(
                        nome=nome,
                        nivel=nivel_enum,
                        data_inicio=datetime.now(),
                        progresso_competencias={},
                        historico_redacoes=[],
                        feedback_acumulado={},
                        ultima_atividade=datetime.now(),
                        total_exercicios=0,
                        medalhas=[]
                    )
                    st.success("Perfil criado com sucesso!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Erro ao criar perfil: {str(e)}")

def show_dashboard_page():
    st.title("📊 Dashboard")
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de progresso por competência
        competencias_data = {
            comp.value: st.session_state.perfil_aluno.progresso_competencias.get(
                comp, ProgressoCompetencia(0, 0, 0, [], [], datetime.now())
            ).nivel
            for comp in CompetenciaModulo
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(competencias_data.keys()),
                y=list(competencias_data.values()),
                marker_color='rgb(26, 118, 255)'
            )
        ])
        
        fig.update_layout(
            title="Progresso por Competência",
            yaxis_title="Nível (0-1)",
            xaxis_title="Competências"
        )
        
        st.plotly_chart(fig)
        
        # Últimas atividades
        st.subheader("📝 Últimas Atividades")
        if st.session_state.perfil_aluno.historico_redacoes:
            for redacao in st.session_state.perfil_aluno.historico_redacoes[-3:]:
                with st.expander(f"Redação: {redacao.tema}"):
                    st.write(f"Data: {redacao.data}")
                    st.write("Notas:")
                    for comp, nota in redacao.notas.items():
                        st.write(f"- {comp.value}: {nota}")
    
    with col2:
        # Estatísticas rápidas
        st.subheader("📈 Estatísticas")
        
        total_redacoes = len(st.session_state.perfil_aluno.historico_redacoes)
        media_geral = sum(
            sum(r.notas.values()) / len(r.notas)
            for r in st.session_state.perfil_aluno.historico_redacoes
        ) / max(total_redacoes, 1)
        
        st.metric("Redações Realizadas", total_redacoes)
        st.metric("Média Geral", f"{media_geral:.1f}")
        st.metric("Exercícios Completados", 
                 st.session_state.perfil_aluno.total_exercicios)
        
        # Próximas metas
        st.subheader("🎯 Próximas Metas")
        metas = gerar_proximas_metas(st.session_state.perfil_aluno)
        for meta in metas:
            st.markdown(f"- {meta}")

def gerar_proximas_metas(perfil: PerfilAluno) -> List[str]:
    """Gera lista de próximas metas baseado no perfil"""
    metas = []
    
    # Identifica competência mais fraca
    comp_mais_fraca = min(
        perfil.progresso_competencias.items(),
        key=lambda x: x[1].nivel
    )[0]
    
    metas.append(f"Melhorar {comp_mais_fraca.value}")
    
    # Adiciona metas baseadas no nível
    if perfil.total_exercicios < 10:
        metas.append("Completar 10 exercícios")
    
    if len(perfil.historico_redacoes) < 3:
        metas.append("Escrever 3 redações completas")
    
    return metas

# Funções auxiliares para análise de redação
async def analisar_redacao_completa(texto: str, tema: str) -> Dict:
    """Realiza análise completa da redação usando todos os módulos"""
    try:
        resultados = {}
        
        # Análise paralela de todas as competências
        tasks = [
            analisar_competencia(CompetenciaModulo.NORMA_CULTA, texto, tema),
            analisar_competencia(CompetenciaModulo.INTERPRETACAO, texto, tema),
            analisar_competencia(CompetenciaModulo.ARGUMENTACAO, texto, tema),
            analisar_competencia(CompetenciaModulo.COESAO, texto, tema),
            analisar_competencia(CompetenciaModulo.PROPOSTA, texto, tema)
        ]
        
        # Aguarda todas as análises
        analises = await asyncio.gather(*tasks)
        
        # Combina resultados
        for comp, analise in zip(CompetenciaModulo, analises):
            resultados[comp] = analise
        
        return resultados
        
    except Exception as e:
        logger.error(f"Erro na análise completa: {e}")
        raise

async def analisar_competencia(comp: CompetenciaModulo, texto: str, tema: str) -> Dict:
    """Analisa uma competência específica"""
    modulo = obter_modulo_competencia(comp)
    
    if comp == CompetenciaModulo.NORMA_CULTA:
        return await modulo.analisar_texto(texto)
    elif comp == CompetenciaModulo.INTERPRETACAO:
        return await modulo.analisar_compreensao(tema, [], texto)
    elif comp == CompetenciaModulo.ARGUMENTACAO:
        return await modulo.analisar_argumentacao(texto, tema)
    elif comp == CompetenciaModulo.COESAO:
        return await modulo.analisar_coesao(texto)
    elif comp == CompetenciaModulo.PROPOSTA:
        return await modulo.analisar_proposta(texto, tema)

def show_nova_redacao_page():
    st.title("📝 Nova Redação")
    
    # Tabs para diferentes etapas
    tabs = st.tabs([
        "✍️ Escrita",
        "🔍 Análise",
        "📋 Revisão"
    ])
    
    with tabs[0]:  # Aba de Escrita
        if "tema_atual" not in st.session_state:
            st.session_state.tema_atual = ""
        if "texto_atual" not in st.session_state:
            st.session_state.texto_atual = ""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tema = st.text_input(
                "Tema da redação:",
                value=st.session_state.tema_atual,
                help="Digite o tema proposto para a redação"
            )
            
            texto = st.text_area(
                "Digite sua redação:",
                value=st.session_state.texto_atual,
                height=400,
                help="Digite seu texto. Separe os parágrafos com uma linha em branco."
            )
            
            if st.button("Analisar Redação") and texto and tema:
                st.session_state.tema_atual = tema
                st.session_state.texto_atual = texto
                
                with st.spinner("Analisando sua redação..."):
                    try:
                        # Executa análise assíncrona
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        resultados = loop.run_until_complete(
                            analisar_redacao_completa(texto, tema)
                        )
                        loop.close()
                        
                        st.session_state.ultima_analise = resultados
                        st.success("Análise concluída! Veja os resultados na aba Análise.")
                        
                        # Salva redação no histórico
                        nova_redacao = Redacao(
                            tema=tema,
                            texto=texto,
                            data=datetime.now(),
                            notas={
                                comp: resultado.get('score_geral', 0)
                                for comp, resultado in resultados.items()
                            },
                            feedback={
                                comp: resultado.get('feedback_detalhado', [])
                                for comp, resultado in resultados.items()
                            }
                        )
                        
                        st.session_state.perfil_aluno.historico_redacoes.append(
                            nova_redacao
                        )
                        
                    except Exception as e:
                        st.error(f"Erro na análise: {str(e)}")
        
        with col2:
            st.markdown("### 💡 Dicas de Escrita")
            
            with st.expander("📌 Estrutura Básica"):
                st.markdown("""
                1. Introdução
                   - Contextualização
                   - Tese clara
                
                2. Desenvolvimento
                   - 2-3 parágrafos
                   - Argumentos sólidos
                   - Exemplos concretos
                
                3. Conclusão
                   - Retomada da tese
                   - Proposta detalhada
                """)
            
            with st.expander("🎯 Critérios de Avaliação"):
                st.markdown("""
                - Competência 1: Norma culta
                - Competência 2: Compreensão da proposta
                - Competência 3: Argumentação
                - Competência 4: Coesão textual
                - Competência 5: Proposta de intervenção
                """)
            
            # Contador de palavras em tempo real
            if texto:
                palavras = len(texto.split())
                st.metric("Palavras", palavras)
                
                if palavras < 2000:
                    st.warning(f"Mínimo recomendado: 2000 palavras")
                elif palavras > 3000:
                    st.warning(f"Máximo recomendado: 3000 palavras")
                else:
                    st.success("Quantidade de palavras adequada!")
    
    with tabs[1]:  # Aba de Análise
        if "ultima_analise" in st.session_state:
            show_analise_redacao(
                st.session_state.ultima_analise,
                st.session_state.texto_atual
            )
        else:
            st.info("Escreva sua redação e clique em Analisar para ver os resultados.")
    
    with tabs[2]:  # Aba de Revisão
        if "ultima_analise" in st.session_state:
            show_revisao_redacao(
                st.session_state.ultima_analise,
                st.session_state.texto_atual
            )
        else:
            st.info("Primeiro faça a análise para ver sugestões de revisão.")

def show_analise_redacao(analise: Dict, texto: str):
    """Mostra a análise detalhada da redação"""
    st.markdown("### 📊 Análise Completa")
    
    # Visão geral das notas
    col1, col2, col3, col4, col5 = st.columns(5)
    colunas = {
        CompetenciaModulo.NORMA_CULTA: col1,
        CompetenciaModulo.INTERPRETACAO: col2,
        CompetenciaModulo.ARGUMENTACAO: col3,
        CompetenciaModulo.COESAO: col4,
        CompetenciaModulo.PROPOSTA: col5
    }
    
    for comp, col in colunas.items():
        with col:
            nota = analise[comp].get('score_geral', 0)
            st.metric(
                f"Comp. {comp.value[-1]}",
                f"{nota:.1f}",
                delta=None if nota >= 160 else f"{160-nota:.1f} para 160"
            )
    
    # Análise por competência
    for comp in CompetenciaModulo:
        with st.expander(f"📝 {comp.value.title()}", expanded=True):
            resultado = analise[comp]
            
            # Texto com marcações
            st.markdown("#### Texto Analisado")
            texto_marcado = texto
            for elem in resultado.get('elementos_destacados', []):
                texto_marcado = destacar_elemento(
                    texto_marcado,
                    elem['texto'],
                    elem['tipo'],
                    elem['comentario']
                )
            st.markdown(texto_marcado, unsafe_allow_html=True)
            
            # Feedback detalhado
            st.markdown("#### 💡 Feedback")
            st.markdown(resultado.get('feedback_detalhado', ''))
            
            # Sugestões de melhoria
            if resultado.get('sugestoes_melhoria'):
                st.markdown("#### ✨ Sugestões de Melhoria")
                for sugestao in resultado['sugestoes_melhoria']:
                    st.markdown(f"- {sugestao}")

def destacar_elemento(texto: str, trecho: str, tipo: str, comentario: str) -> str:
    """Destaca elementos no texto com cores e tooltips"""
    cores = {
        'erro_gramatical': '#ff6b6b',
        'conectivo': '#4dabf7',
        'argumento': '#69db7c',
        'tese': '#ffd43b',
        'repertorio': '#da77f2',
        'proposta': '#4c6ef5',
        'problema': '#ff8787',
        'destaque_positivo': '#51cf66',
        'destaque_negativo': '#ff6b6b'
    }
    
    cor = cores.get(tipo, '#868e96')
    return texto.replace(
        trecho,
        f'<span style="background-color: {cor}33; border-bottom: 2px solid {cor}; '
        f'cursor: help;" title="{comentario}">{trecho}</span>'
    )

def mostrar_pagina_competencia(competencia: CompetenciaModulo):
    st.title(f"📝 {competencia.value.title()}")
    
    # Tabs principais
    tab_aprenda, tab_pratique, tab_analise = st.tabs([
        "📚 Aprenda",
        "✍️ Pratique",
        "🔍 Analise"
    ])
    
    with tab_aprenda:
        mostrar_conteudo_aprendizagem(competencia)
    
    with tab_pratique:
        mostrar_exercicios_competencia(competencia)
    
    with tab_analise:
        mostrar_analise_competencia(competencia)

def mostrar_conteudo_aprendizagem(competencia: CompetenciaModulo):
    """Mostra conteúdo didático da competência"""
    materiais = obter_material_competencia(competencia)
    
    # Visão geral
    st.markdown("### 📋 Visão Geral")
    st.markdown(materiais['visao_geral'])
    
    # Critérios de avaliação
    with st.expander("🎯 Critérios de Avaliação", expanded=True):
        for criterio in materiais['criterios']:
            st.markdown(f"- **{criterio['nome']}**: {criterio['descricao']}")
    
    # Exemplos comentados
    with st.expander("📝 Exemplos Comentados", expanded=True):
        for exemplo in materiais['exemplos']:
            st.markdown(f"#### {exemplo['titulo']}")
            texto_marcado = exemplo['texto']
            for marcacao in exemplo['marcacoes']:
                texto_marcado = destacar_elemento(
                    texto_marcado,
                    marcacao['trecho'],
                    marcacao['tipo'],
                    marcacao['comentario']
                )
            st.markdown(texto_marcado, unsafe_allow_html=True)
            st.markdown(f"**Análise**: {exemplo['analise']}")
    
    # Dicas práticas
    with st.expander("💡 Dicas Práticas", expanded=True):
        for dica in materiais['dicas']:
            st.markdown(f"- {dica}")

def mostrar_exercicios_competencia(competencia: CompetenciaModulo):
    """Mostra exercícios específicos da competência"""
    if "exercicio_atual" not in st.session_state:
        st.session_state.exercicio_atual = None
    
    # Seleção de foco
    focos_disponiveis = obter_focos_competencia(competencia)
    foco_selecionado = st.multiselect(
        "Escolha os aspectos que deseja praticar:",
        focos_disponiveis
    )
    
    if foco_selecionado:
        if st.button("Gerar Novo Exercício"):
            with st.spinner("Gerando exercício..."):
                try:
                    modulo = obter_modulo_competencia(competencia)
                    exercicios = asyncio.run(modulo.gerar_exercicios(
                        st.session_state.perfil_aluno.nivel,
                        foco_selecionado
                    ))
                    if exercicios:
                        st.session_state.exercicio_atual = exercicios[0]
                except Exception as e:
                    st.error(f"Erro ao gerar exercício: {str(e)}")
        
        if st.session_state.exercicio_atual:
            mostrar_exercicio(st.session_state.exercicio_atual)

def mostrar_exercicio(exercicio: ExercicioRedacao):
    """Mostra um exercício específico"""
    st.markdown(f"### {exercicio.tipo.title()}")
    
    # Enunciado e instruções
    st.markdown(f"**Enunciado**: {exercicio.enunciado}")
    
    with st.expander("📋 Instruções", expanded=True):
        for i, instrucao in enumerate(exercicio.instrucoes, 1):
            st.markdown(f"{i}. {instrucao}")
    
    # Área de resposta
    resposta = st.text_area(
        "Sua resposta:",
        height=200,
        help="Digite sua resposta aqui"
    )
    
    if st.button("Verificar"):
        with st.spinner("Analisando sua resposta..."):
            try:
                modulo = obter_modulo_competencia(
                    CompetenciaModulo[exercicio.tipo.upper()]
                )
                resultado = asyncio.run(
                    modulo.analisar_texto(resposta)
                )
                mostrar_feedback_exercicio(resultado, exercicio)
                
                # Atualiza progresso
                atualizar_progresso_exercicio(
                    exercicio.tipo,
                    resultado.get('score_geral', 0)
                )
                
            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
    
    # Dicas
    if exercicio.dicas:
        with st.expander("💡 Dicas"):
            for dica in exercicio.dicas:
                st.markdown(f"- {dica}")

def mostrar_feedback_exercicio(resultado: Dict, exercicio: ExercicioRedacao):
    """Mostra feedback detalhado do exercício"""
    st.markdown("### 📊 Resultado")
    
    # Score geral
    score = resultado.get('score_geral', 0)
    st.progress(score/200)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pontuação", f"{score:.1f}/200")
    with col2:
        if score >= 160:
            st.success("Excelente! Continue assim!")
        elif score >= 120:
            st.info("Bom trabalho! Pode melhorar ainda mais.")
        else:
            st.warning("Continue praticando para melhorar.")
    
    # Feedback detalhado
    if 'feedback_detalhado' in resultado:
        st.markdown("### 💭 Feedback Detalhado")
        st.markdown(resultado['feedback_detalhado'])
    
    # Sugestões de melhoria
    if 'sugestoes_melhoria' in resultado:
        st.markdown("### ✨ Sugestões de Melhoria")
        for sugestao in resultado['sugestoes_melhoria']:
            st.markdown(f"- {sugestao}")
    
    # Exemplo de resposta
    if exercicio.exemplo_resposta:
        with st.expander("📝 Exemplo de Resposta"):
            st.markdown(exercicio.exemplo_resposta)

def mostrar_analise_competencia(competencia: CompetenciaModulo):
    """Interface para análise específica de uma competência"""
    st.markdown("### 🔍 Análise Específica")
    
    # Área de texto para análise
    texto = st.text_area(
        "Digite o texto para análise:",
        height=200,
        help=f"Cole aqui o trecho que deseja analisar quanto à {competencia.value}"
    )
    
    if texto:
        tema = st.text_input(
            "Tema (se aplicável):",
            help="Digite o tema para contexto da análise"
        )
        
        if st.button("Analisar"):
            with st.spinner("Analisando..."):
                try:
                    modulo = obter_modulo_competencia(competencia)
                    resultado = asyncio.run(
                        analisar_competencia(competencia, texto, tema)
                    )
                    mostrar_resultado_analise(resultado, texto, competencia)
                except Exception as e:
                    st.error(f"Erro na análise: {str(e)}")

def mostrar_resultado_analise(resultado: Dict, texto: str, competencia: CompetenciaModulo):
    """Mostra o resultado da análise de forma detalhada"""
    # Score e avaliação geral
    st.markdown("### 📊 Resultado da Análise")
    
    col1, col2 = st.columns(2)
    with col1:
        score = resultado.get('score_geral', 0)
        st.metric(
            "Pontuação",
            f"{score}/200",
            delta=f"{score-160}" if score < 160 else "Meta atingida!"
        )
    
    with col2:
        nivel_analise = "Avançado" if score >= 160 else \
                       "Intermediário" if score >= 120 else "Básico"
        st.info(f"Nível: {nivel_analise}")
    
    # Texto com marcações
    st.markdown("### 📝 Texto Analisado")
    texto_marcado = texto
    for elem in resultado.get('elementos_destacados', []):
        texto_marcado = destacar_elemento(
            texto_marcado,
            elem['texto'],
            elem['tipo'],
            elem['comentario']
        )
    st.markdown(texto_marcado, unsafe_allow_html=True)
    
    # Análise específica por competência
    if competencia == CompetenciaModulo.NORMA_CULTA:
        mostrar_analise_norma_culta(resultado)
    elif competencia == CompetenciaModulo.INTERPRETACAO:
        mostrar_analise_interpretacao(resultado)
    elif competencia == CompetenciaModulo.ARGUMENTACAO:
        mostrar_analise_argumentacao(resultado)
    elif competencia == CompetenciaModulo.COESAO:
        mostrar_analise_coesao(resultado)
    elif competencia == CompetenciaModulo.PROPOSTA:
        mostrar_analise_proposta(resultado)

def mostrar_analise_norma_culta(resultado: Dict):
    """Mostra análise específica da competência 1"""
    # Erros gramaticais
    if 'erros_gramaticais' in resultado:
        with st.expander("🔍 Erros Gramaticais", expanded=True):
            for erro in resultado['erros_gramaticais']:
                st.markdown(f"""
                **Tipo**: {erro['tipo']}  
                **Trecho**: "{erro['trecho']}"  
                **Correção**: {erro['correcao']}  
                **Explicação**: {erro['explicacao']}
                ---
                """)
    
    # Adequação vocabular
    if 'adequacao_vocabular' in resultado:
        with st.expander("📚 Adequação Vocabular"):
            adq = resultado['adequacao_vocabular']
            st.metric("Nível de Formalidade", f"{adq['nivel_formalidade']}/5")
            
            if adq['problemas_identificados']:
                st.markdown("**Problemas Identificados:**")
                for prob in adq['problemas_identificados']:
                    st.markdown(f"- {prob}")
            
            if adq['sugestoes_melhoria']:
                st.markdown("**Sugestões de Melhoria:**")
                for sug in adq['sugestoes_melhoria']:
                    st.markdown(f"- {sug}")

def mostrar_analise_argumentacao(resultado: Dict):
    """Mostra análise específica da competência 3"""
    # Estrutura argumentativa
    if 'estrutura_argumentativa' in resultado:
        with st.expander("🎯 Estrutura Argumentativa", expanded=True):
            est = resultado['estrutura_argumentativa']
            
            # Tese
            st.markdown("#### 📌 Tese")
            st.markdown(f"""
            **Identificada**: {est['tese']['texto']}  
            **Clareza**: {est['tese']['clareza']}/200  
            **Desenvolvimento**: {est['tese']['desenvolvimento']}
            """)
            
            # Argumentos
            st.markdown("#### 💡 Argumentos")
            for arg in est['argumentos']:
                st.markdown(f"""
                **Tipo**: {arg['tipo']}  
                **Força**: {arg['forca']}/200  
                **Desenvolvimento**: {arg['desenvolvimento']}  
                **Sugestões**: {', '.join(arg['sugestoes'])}
                ---
                """)
    
    # Repertório sociocultural
    if 'repertorio_sociocultural' in resultado:
        with st.expander("📚 Repertório Sociocultural"):
            rep = resultado['repertorio_sociocultural']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pertinência", f"{rep['pertinencia']}/200")
            with col2:
                st.metric("Qualidade de Uso", f"{rep['qualidade_uso']}/200")
            
            if rep['areas_presentes']:
                st.markdown("**Áreas do Conhecimento Utilizadas:**")
                for area in rep['areas_presentes']:
                    st.markdown(f"- {area}")
            
            if rep['sugestoes_ampliacao']:
                st.markdown("**Sugestões de Ampliação:**")
                for sug in rep['sugestoes_ampliacao']:
                    st.markdown(f"- {sug}")

def atualizar_progresso(
    perfil: PerfilAluno,
    competencia: CompetenciaModulo,
    score: float,
    feedback: List[str]
):
    """Atualiza o progresso do aluno em uma competência"""
    if competencia not in perfil.progresso_competencias:
        perfil.progresso_competencias[competencia] = ProgressoCompetencia(
            nivel=0.0,
            exercicios_feitos=0,
            ultima_avaliacao=0.0,
            pontos_fortes=[],
            pontos_fracos=[],
            data_atualizacao=datetime.now()
        )
    
    progresso = perfil.progresso_competencias[competencia]
    
    # Atualiza nível (média móvel)
    alpha = 0.3  # peso para nova avaliação
    progresso.nivel = (1 - alpha) * progresso.nivel + alpha * (score / 200)
    
    # Atualiza última avaliação
    progresso.ultima_avaliacao = score / 200
    progresso.data_atualizacao = datetime.now()
    
    # Atualiza feedback acumulado
    if competencia not in perfil.feedback_acumulado:
        perfil.feedback_acumulado[competencia] = []
    
    perfil.feedback_acumulado[competencia].extend(feedback)
    
    # Mantém apenas os últimos 10 feedbacks
    perfil.feedback_acumulado[competencia] = \
        perfil.feedback_acumulado[competencia][-10:]
    
    # Verifica conquistas
    verificar_conquistas(perfil, competencia, score)

def verificar_conquistas(
    perfil: PerfilAluno,
    competencia: CompetenciaModulo,
    score: float
):
    """Verifica e atribui conquistas baseadas no desempenho"""
    if not perfil.medalhas:
        perfil.medalhas = []
    
    # Conquistas por competência
    if score >= 180:
        medalha = f"Mestre em {competencia.value}"
        if medalha not in perfil.medalhas:
            perfil.medalhas.append(medalha)
            st.balloons()
            st.success(f"🏆 Nova conquista: {medalha}!")
    
    elif score >= 160:
        medalha = f"Especialista em {competencia.value}"
        if medalha not in perfil.medalhas:
            perfil.medalhas.append(medalha)
            st.success(f"🎖️ Nova conquista: {medalha}!")
    
    # Conquistas gerais
    if len(perfil.historico_redacoes) == 10:
        medalha = "Escritor Dedicado"
        if medalha not in perfil.medalhas:
            perfil.medalhas.append(medalha)
            st.success(f"📝 Nova conquista: {medalha}!")
    
    if perfil.total_exercicios >= 50:
        medalha = "Mestre dos Exercícios"
        if medalha not in perfil.medalhas:
            perfil.medalhas.append(medalha)
            st.success(f"✨ Nova conquista: {medalha}!")


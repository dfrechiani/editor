import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import language_tool_python
import concurrent.futures

# Configuração da página
st.set_page_config(
    page_title="Editor Interativo de Redação ENEM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Classes de dados básicas
@dataclass
class AnaliseElementos:
    presentes: List[str]
    ausentes: List[str]
    score: float
    sugestoes: List[str]

@dataclass
class CorrecaoGramatical:
    texto_original: str
    sugestoes: List[dict]
    total_erros: int
    categorias_erros: dict
    texto_corrigido: Optional[str] = None

@dataclass
class ConectivoAnalise:
    texto: str
    tipo: str
    posicao: Tuple[int, int]
    frequencia: int

@dataclass
class ArgumentoAnalise:
    texto: str
    tipo: str  # "A1" ou "A2"
    posicao: Tuple[int, int]  # início e fim do argumento no texto
    score: float
    feedback: List[str]
    tratamento: dict  # detalhes do tratamento conforme critérios ENEM

@dataclass
class AnaliseConectivos:
    conectivos: List[ConectivoAnalise]
    estatisticas: Dict[str, int]
    repeticoes: Dict[str, int]
    score: float
    feedback: List[str]

@dataclass
class AnaliseParagrafo:
    tipo: str
    texto: str
    elementos: AnaliseElementos
    feedback: List[str]
    correcao_gramatical: Optional[CorrecaoGramatical] = None
    argumentos: Optional[List[ArgumentoAnalise]] = None
    analise_conectivos: Optional[AnaliseConectivos] = None
    tempo_analise: float = 0.0

# Configuração da API e modelos
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

class ModeloAnalise(str, Enum):
    RAPIDO = "gpt-3.5-turbo-1106"

# Configurações de processamento
MAX_WORKERS = 3
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Configurações de timeout e retry
API_TIMEOUT = 10.0
MAX_RETRIES = 2
MIN_PALAVRAS_IA = 20
MAX_PALAVRAS_IA = 300

# Cache com TTL para melhor performance
class CacheAnalise:
    def __init__(self, max_size: int = 50, ttl_seconds: int = 180):
        self.cache: Dict[str, Tuple[AnaliseElementos, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, texto: str, tipo: str) -> Optional[AnaliseElementos]:
        try:
            chave = f"{tipo}:{hash(texto)}"
            if chave in self.cache:
                analise, timestamp = self.cache[chave]
                if (datetime.now() - timestamp).seconds < self.ttl_seconds:
                    return analise
                del self.cache[chave]
        except Exception as e:
            logger.error(f"Erro ao acessar cache: {e}")
        return None
    
    def set(self, texto: str, tipo: str, analise: AnaliseElementos) -> None:
        try:
            if len(self.cache) >= self.max_size:
                agora = datetime.now()
                expirados = [
                    k for k, (_, t) in self.cache.items() 
                    if (agora - t).seconds >= self.ttl_seconds
                ]
                for k in expirados:
                    del self.cache[k]
                
                if len(self.cache) >= self.max_size:
                    mais_antigo = min(self.cache.items(), key=lambda x: x[1][1])
                    del self.cache[mais_antigo[0]]
            
            self.cache[f"{tipo}:{hash(texto)}"] = (analise, datetime.now())
        except Exception as e:
            logger.error(f"Erro ao definir cache: {e}")

class VerificadorGramatical:
    def __init__(self):
        try:
            self.tool = language_tool_python.LanguageToolPublicAPI('pt-BR')
            self.initialized = True
        except Exception as e:
            logger.error(f"Erro ao inicializar LanguageTool: {e}")
            self.initialized = False
            
    def verificar_texto(self, texto: str) -> CorrecaoGramatical:
        if not self.initialized:
            return CorrecaoGramatical(
                texto_original=texto,
                sugestoes=[],
                total_erros=0,
                categorias_erros={},
                texto_corrigido=None
            )
            
        try:
            matches = self.tool.check(texto)
            
            sugestoes = []
            categorias_erros = {}
            texto_corrigido = texto
            
            for match in matches:
                categoria = match.category
                if categoria not in categorias_erros:
                    categorias_erros[categoria] = 0
                categorias_erros[categoria] += 1
                
                sugestao = {
                    'erro': texto[match.offset:match.offset + match.errorLength],
                    'sugestoes': match.replacements,
                    'mensagem': match.message,
                    'categoria': categoria,
                    'contexto': self._get_context(texto, match.offset, match.errorLength),
                    'posicao': match.offset
                }
                sugestoes.append(sugestao)
                
                if match.replacements:
                    texto_corrigido = texto_corrigido.replace(
                        texto[match.offset:match.offset + match.errorLength],
                        match.replacements[0]
                    )
            
            return CorrecaoGramatical(
                texto_original=texto,
                sugestoes=sugestoes,
                total_erros=len(sugestoes),
                categorias_erros=categorias_erros,
                texto_corrigido=texto_corrigido
            )
            
        except Exception as e:
            logger.error(f"Erro na verificação gramatical: {e}")
            return CorrecaoGramatical(
                texto_original=texto,
                sugestoes=[],
                total_erros=0,
                categorias_erros={},
                texto_corrigido=None
            )
    
    def _get_context(self, texto: str, offset: int, length: int, context_size: int = 40) -> str:
        start = max(0, offset - context_size)
        end = min(len(texto), offset + length + context_size)
        
        context = texto[start:end]
        if start > 0:
            context = f"...{context}"
        if end < len(texto):
            context = f"{context}..."
            
        return context

class AnalisadorArgumentos:
    def __init__(self):
        self.marcadores_a1_a2 = {
            "introducao": [
                "primeiro", "inicialmente", "primeiramente", "por um lado",
                "além disso", "ademais", "outrossim", "por outro lado",
                "soma-se a isso", "adiciona-se a isso"
            ],
            "desenvolvimento": [
                "com efeito", "de fato", "certamente", "evidentemente",
                "sob essa perspectiva", "nesse sentido", "diante disso",
                "à luz dessa", "sob esse aspecto"
            ]
        }
        
        self.criterios_tratamento = {
            "desenvolvimento1": {
                "argumentacao": ["citação direta", "dados estatísticos", "fatos históricos"],
                "justificativa": ["explicação", "exemplificação", "causa-consequência"],
                "repertorio": ["obras literárias", "filosofia", "sociologia", "atualidades"]
            },
            "desenvolvimento2": {
                "argumentacao": ["comparação", "analogia", "contraposição"],
                "justificativa": ["contextualização", "análise", "relação"],
                "repertorio": ["cinema", "artes", "ciências", "tecnologia"]
            }
        }

    def identificar_argumentos(self, texto: str, tipo_paragrafo: str) -> List[ArgumentoAnalise]:
        argumentos = []
        texto_lower = texto.lower()
        
        marcadores = self.marcadores_a1_a2.get(
            "introducao" if tipo_paragrafo == "introducao" else "desenvolvimento",
            []
        )
        
        for i, marcador in enumerate(marcadores):
            pos = texto_lower.find(marcador)
            if pos >= 0:
                fim = len(texto)
                for next_marcador in marcadores[i+1:]:
                    next_pos = texto_lower.find(next_marcador)
                    if next_pos > pos:
                        fim = next_pos
                        break
                
                tipo_arg = "A1" if not argumentos else "A2"
                
                argumento = ArgumentoAnalise(
                    texto=texto[pos:fim].strip(),
                    tipo=tipo_arg,
                    posicao=(pos, fim),
                    score=self.avaliar_argumento(texto[pos:fim], tipo_paragrafo),
                    feedback=self.gerar_feedback_argumento(texto[pos:fim], tipo_paragrafo),
                    tratamento=self.analisar_tratamento(texto[pos:fim], tipo_paragrafo)
                )
                argumentos.append(argumento)
                
                if len(argumentos) >= 2:
                    break
        
        return argumentos
    def avaliar_argumento(self, texto: str, tipo_paragrafo: str) -> float:
        score = 0.0
        criterios = self.criterios_tratamento.get(tipo_paragrafo, {})
        
        for categoria, elementos in criterios.items():
            for elemento in elementos:
                if any(marker in texto.lower() for marker in self.get_markers_for_element(elemento)):
                    score += 0.25
        
        return min(1.0, score)

    def analisar_tratamento(self, texto: str, tipo_paragrafo: str) -> dict:
        tratamento = {}
        criterios = self.criterios_tratamento.get(tipo_paragrafo, {})
        
        for categoria, elementos in criterios.items():
            presentes = []
            for elemento in elementos:
                if any(marker in texto.lower() for marker in self.get_markers_for_element(elemento)):
                    presentes.append(elemento)
            if presentes:
                tratamento[categoria] = presentes
        
        return tratamento

    def gerar_feedback_argumento(self, texto: str, tipo_paragrafo: str) -> List[str]:
        feedback = []
        tratamento = self.analisar_tratamento(texto, tipo_paragrafo)
        
        for categoria, elementos in tratamento.items():
            if elementos:
                feedback.append(f"✓ Bom uso de {categoria} com {', '.join(elementos)}")
        
        criterios = self.criterios_tratamento.get(tipo_paragrafo, {})
        for categoria, elementos in criterios.items():
            if categoria not in tratamento:
                feedback.append(f"💡 Sugestão: Inclua {categoria} usando {', '.join(elementos[:2])}")
        
        return feedback

    def get_markers_for_element(self, elemento: str) -> List[str]:
        markers_map = {
            "citação direta": ["afirma", "declara", "segundo", "conforme"],
            "dados estatísticos": ["porcentagem", "número", "taxa", "índice"],
            "fatos históricos": ["história", "período", "época", "durante"],
            "explicação": ["porque", "pois", "uma vez que", "já que"],
            "exemplificação": ["exemplo", "como", "tal qual", "assim como"],
            "causa-consequência": ["causa", "consequência", "efeito", "resultado"],
            "obras literárias": ["livro", "obra", "romance", "autor"],
            "filosofia": ["filósofo", "pensamento", "conceito", "teoria"],
            "sociologia": ["sociedade", "social", "sociólogo", "fenômeno"],
            "atualidades": ["atual", "recente", "hoje", "contemporâneo"],
            "comparação": ["mais", "menos", "tanto quanto", "assim como"],
            "analogia": ["similar", "semelhante", "análogo", "como"],
            "contraposição": ["contrário", "oposto", "diferente", "enquanto"],
            "contextualização": ["contexto", "cenário", "situação", "realidade"],
            "análise": ["analisar", "examinar", "verificar", "observar"],
            "relação": ["relacionar", "conectar", "vincular", "ligar"]
        }
        return markers_map.get(elemento, [elemento.lower()])

class AnalisadorConectivos:
    def __init__(self):
        self.conectivos_por_tipo = {
            "aditivos": [
                "além disso", "ademais", "também", "e", "outrossim",
                "não apenas... mas também", "inclusive", "ainda", 
                "nem", "não só... mas também"
            ],
            "adversativos": [
                "mas", "porém", "contudo", "entretanto", "no entanto",
                "todavia", "não obstante", "apesar de", "embora", 
                "ainda que", "mesmo que", "posto que"
            ],
            "conclusivos": [
                "portanto", "logo", "assim", "dessa forma", "por isso",
                "consequentemente", "por conseguinte", "então", 
                "destarte", "desse modo", "sendo assim"
            ],
            "explicativos": [
                "pois", "porque", "já que", "visto que", "uma vez que",
                "porquanto", "posto que", "tendo em vista que", 
                "haja vista que", "considerando que"
            ],
            "sequenciais": [
                "primeiramente", "em seguida", "por fim", "depois",
                "anteriormente", "posteriormente", "finalmente",
                "em primeiro lugar", "em segundo lugar", "por último"
            ],
            "comparativos": [
                "assim como", "tal qual", "tanto quanto", "como",
                "da mesma forma", "igualmente", "similarmente",
                "do mesmo modo", "semelhantemente"
            ],
            "enfáticos": [
                "com efeito", "de fato", "realmente", "evidentemente",
                "naturalmente", "decerto", "certamente", "sobretudo",
                "principalmente", "especialmente"
            ]
        }

    def identificar_conectivos(self, texto: str) -> AnaliseConectivos:
        conectivos_encontrados = []
        estatisticas = {}
        repeticoes = {}
        texto_lower = texto.lower()
        
        for tipo, lista_conectivos in self.conectivos_por_tipo.items():
            estatisticas[tipo] = 0
            
            for conectivo in lista_conectivos:
                posicoes = self._encontrar_todas_ocorrencias(texto_lower, conectivo)
                
                if posicoes:
                    frequencia = len(posicoes)
                    estatisticas[tipo] += frequencia
                    
                    if frequencia > 1:
                        repeticoes[conectivo] = frequencia
                    
                    for pos in posicoes:
                        conectivos_encontrados.append(ConectivoAnalise(
                            texto=texto[pos:pos + len(conectivo)],
                            tipo=tipo,
                            posicao=(pos, pos + len(conectivo)),
                            frequencia=frequencia
                        ))
        
        score = self._calcular_score(estatisticas, repeticoes)
        feedback = self._gerar_feedback(estatisticas, repeticoes)
        
        return AnaliseConectivos(
            conectivos=conectivos_encontrados,
            estatisticas=estatisticas,
            repeticoes=repeticoes,
            score=score,
            feedback=feedback
        )

    def _encontrar_todas_ocorrencias(self, texto: str, substring: str) -> List[int]:
        posicoes = []
        pos = texto.find(substring)
        while pos != -1:
            posicoes.append(pos)
            pos = texto.find(substring, pos + 1)
        return posicoes

    def _calcular_score(self, estatisticas: Dict[str, int], repeticoes: Dict[str, int]) -> float:
        tipos_usados = sum(1 for count in estatisticas.values() if count > 0)
        total_conectivos = sum(estatisticas.values())
        
        if total_conectivos == 0:
            return 0.0
        
        # Base score pela variedade de tipos
        score = tipos_usados / len(self.conectivos_por_tipo) * 0.5
        
        # Adiciona pontuação pela quantidade adequada
        quantidade_ideal = 3  # Número ideal de conectivos por tipo
        distribuicao = sum(
            min(count / quantidade_ideal, 1.0) 
            for count in estatisticas.values()
        ) / len(self.conectivos_por_tipo)
        score += distribuicao * 0.3
        
        # Penaliza repetições excessivas
        penalidade_repeticoes = len(repeticoes) * 0.05
        score = max(0.0, score - penalidade_repeticoes)
        
        return min(1.0, score)

    def _gerar_feedback(self, estatisticas: Dict[str, int], repeticoes: Dict[str, int]) -> List[str]:
        feedback = []
        
        tipos_usados = sum(1 for count in estatisticas.values() if count > 0)
        if tipos_usados >= 5:
            feedback.append("✨ Excelente variedade de conectivos!")
        elif tipos_usados >= 3:
            feedback.append("✓ Boa variedade de conectivos.")
        else:
            feedback.append("💡 Procure utilizar mais tipos diferentes de conectivos.")
        
        for tipo, count in estatisticas.items():
            if count == 0:
                feedback.append(f"📌 Sugestão: Considere usar conectivos {tipo}.")
            elif count > 4:
                feedback.append(f"⚠️ Uso frequente de conectivos {tipo}.")
        
        if repeticoes:
            feedback.append("🔄 Conectivos repetidos:")
            for conectivo, freq in repeticoes.items():
                feedback.append(f"  • '{conectivo}' usado {freq} vezes")
        
        return feedback

# Funções de análise e display
def detectar_tipo_paragrafo(texto: str, posicao: Optional[int] = None) -> str:
    try:
        if posicao is not None:
            if posicao == 0:
                return "introducao"
            elif posicao in [1, 2]:
                return f"desenvolvimento{posicao}"
            elif posicao == 3:
                return "conclusao"
        
        texto_lower = texto.lower()
        
        # Verifica conclusão primeiro (mais distintivo)
        if any(marker in texto_lower for marker in MARKERS["conclusao"]["agente"]) or \
           any(marker in texto_lower for marker in MARKERS["conclusao"]["acao"]):
            return "conclusao"
            
        # Verifica introdução
        if any(marker in texto_lower for marker in MARKERS["introducao"]["contexto"]) or \
           any(marker in texto_lower for marker in MARKERS["introducao"]["tese"]):
            return "introducao"
        
        # Default para desenvolvimento
        return "desenvolvimento1"
        
    except Exception as e:
        logger.error(f"Erro na detecção do tipo de parágrafo: {e}")
        return "desenvolvimento1"

def destacar_argumentos(texto: str, argumentos: List[ArgumentoAnalise]) -> str:
    cores = {
        "A1": "#4CAF50",  # Verde
        "A2": "#2196F3"   # Azul
    }
    
    argumentos_ordenados = sorted(argumentos, key=lambda x: x.posicao[0], reverse=True)
    texto_destacado = texto
    
    for arg in argumentos_ordenados:
        inicio, fim = arg.posicao
        texto_original = texto_destacado[inicio:fim]
        texto_destacado = (
            texto_destacado[:inicio] +
            f'<span style="background-color: {cores[arg.tipo]}33; '
            f'border-left: 3px solid {cores[arg.tipo]}; '
            f'padding: 2px 5px; margin: 2px 0; display: inline-block;" '
            f'title="Qualidade do argumento: {int(arg.score * 100)}%">'
            f'{texto_original}</span>' +
            texto_destacado[fim:]
        )
    
    return texto_destacado

def destacar_conectivos(texto: str, analise: AnaliseConectivos) -> str:
    cores_tipo = {
        "aditivos": "#9C27B0",      # Roxo
        "adversativos": "#FF5722",   # Laranja
        "conclusivos": "#673AB7",    # Roxo escuro
        "explicativos": "#009688",   # Verde água
        "sequenciais": "#795548",    # Marrom
        "comparativos": "#607D8B",   # Azul acinzentado
        "enfáticos": "#FF9800"       # Laranja claro
    }
    
    conectivos_ordenados = sorted(
        analise.conectivos,
        key=lambda x: x.posicao[0],
        reverse=True
    )
    
    texto_destacado = texto
    for conectivo in conectivos_ordenados:
        inicio, fim = conectivo.posicao
        texto_original = texto_destacado[inicio:fim]
        
        tooltip = (
            f"Tipo: {conectivo.tipo}\n"
            f"Frequência: {conectivo.frequencia}x"
        )
        
        texto_destacado = (
            texto_destacado[:inicio] +
            f'<span style="background-color: {cores_tipo[conectivo.tipo]}33; '
            f'border-bottom: 2px dashed {cores_tipo[conectivo.tipo]}; '
            f'padding: 0 2px; cursor: help;" '
            f'title="{tooltip}">'
            f'{texto_original}</span>' +
            texto_destacado[fim:]
        )
    
    return texto_destacado

def marcar_erros_no_texto(texto: str, correcoes: CorrecaoGramatical) -> str:
    if not correcoes or not correcoes.sugestoes:
        return texto
    
    sugestoes_ordenadas = sorted(
        correcoes.sugestoes,
        key=lambda x: x['posicao'],
        reverse=True
    )
    
    texto_marcado = texto
    for sugestao in sugestoes_ordenadas:
        erro = sugestao['erro']
        posicao = sugestao['posicao']
        sugestao_texto = sugestao['sugestoes'][0] if sugestao['sugestoes'] else ''
        marcacao = f'<span style="background-color: rgba(255, 107, 107, 0.3); border-bottom: 2px dashed #ff6b6b; cursor: help;" title="Sugestão: {sugestao_texto}">{erro}</span>'
        texto_marcado = (
            texto_marcado[:posicao] +
            marcacao +
            texto_marcado[posicao + len(erro):]
        )
    
    return texto_marcado

def mostrar_analise_argumentos(argumentos: List[ArgumentoAnalise]):
    if not argumentos:
        st.info("Nenhum argumento identificado neste parágrafo.")
        return
        
    for arg in argumentos:
        with st.expander(f"📝 Análise do {arg.tipo}", expanded=True):
            # Score do argumento
            st.markdown(
                f"""<div style='
                    background-color: #1a472a;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                '>
                    <h4>Qualidade do Argumento: {int(arg.score * 100)}%</h4>
                </div>""",
                unsafe_allow_html=True
            )
            
            # Tratamento do argumento
            if arg.tratamento:
                st.markdown("#### 🎯 Elementos Identificados")
                for categoria, elementos in arg.tratamento.items():
                    st.markdown(
                        f"""<div style='
                            background-color: #262730;
                            padding: 10px;
                            border-radius: 5px;
                            margin: 5px 0;
                        '>
                            <p><strong>{categoria.title()}:</strong></p>
                            <p>{', '.join(elementos)}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )
            
            # Feedback
            st.markdown("#### 💡 Feedback")
            for fb in arg.feedback:
                st.markdown(
                    f"""<div style='
                        background-color: #1e2a3a;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 5px 0;
                    '>{fb}</div>""",
                    unsafe_allow_html=True
                )

def mostrar_analise_conectivos(analise: AnaliseConectivos):
    st.markdown("### 🔄 Análise de Conectivos")
    
    # Métricas principais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_conectivos = sum(analise.estatisticas.values())
        st.metric("Total de Conectivos", total_conectivos)
    
    with col2:
        tipos_usados = sum(1 for count in analise.estatisticas.values() if count > 0)
        st.metric("Tipos Diferentes", tipos_usados)
    
    with col3:
        st.metric(
            "Qualidade do Uso",
            f"{int(analise.score * 100)}%",
            delta="Bom" if analise.score >= 0.7 else None
        )
    
    # Distribuição por tipo
    st.markdown("#### 📊 Distribuição por Tipo")
    
    # Dados para o gráfico
    dados_grafico = [
        {"tipo": tipo, "quantidade": qtd}
        for tipo, qtd in analise.estatisticas.items()
        if qtd > 0  # Só mostra tipos que foram usados
    ]
    
    if dados_grafico:
        st.bar_chart(dados_grafico)
    
    # Análise de repetições
    if analise.repeticoes:
        st.markdown("#### 🔄 Conectivos Repetidos")
        for conectivo, freq in analise.repeticoes.items():
            st.markdown(
                f"""<div style='
                    background-color: #1e2a3a;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                '>
                    <p><strong>'{conectivo}'</strong> usado {freq} vezes</p>
                </div>""",
                unsafe_allow_html=True
            )
    
    # Feedback e sugestões
    st.markdown("#### 💡 Feedback e Sugestões")
    for fb in analise.feedback:
        bg_color = "#1a472a" if "✨" in fb or "✓" in fb else "#262730"
        if "⚠️" in fb:
            bg_color = "#4a1919"
        
        st.markdown(
            f"""<div style='
                background-color: {bg_color};
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
            '>{fb}</div>""",
            unsafe_allow_html=True
        )

def mostrar_legenda_conectivos():
    st.markdown("#### 🎨 Legenda de Conectivos")
    
    cores = {
        "Aditivos": "#9C27B0",
        "Adversativos": "#FF5722",
        "Conclusivos": "#673AB7",
        "Explicativos": "#009688",
        "Sequenciais": "#795548",
        "Comparativos": "#607D8B",
        "Enfáticos": "#FF9800"
    }
    
    legenda_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    
    for tipo, cor in cores.items():
        legenda_html += f"""
            <div style='
                display: flex;
                align-items: center;
                margin: 5px;
                background-color: {cor}33;
                padding: 5px 10px;
                border-radius: 3px;
                border-bottom: 2px dashed {cor};
            '>
                <span>{tipo}</span>
            </div>
        """
    
    legenda_html += "</div>"
    st.markdown(legenda_html, unsafe_allow_html=True)

def mostrar_correcoes_gramaticais(correcao: CorrecaoGramatical):
    if not correcao or not correcao.sugestoes:
        st.success("✓ Não foram encontrados erros gramaticais significativos.")
        return
        
    st.markdown("### 📝 Correções Gramaticais")
    
    # Resumo dos erros
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de correções sugeridas", correcao.total_erros)
    with col2:
        categorias = sorted(
            correcao.categorias_erros.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if categorias:
            st.markdown("**Principais categorias:**")
            for categoria, count in categorias[:3]:
                st.markdown(f"- {categoria}: {count}")
    
    # Detalhamento das correções
    with st.expander("Ver todas as correções sugeridas", expanded=True):
        for i, sugestao in enumerate(correcao.sugestoes, 1):
            st.markdown(
                f"""<div style='
                    background-color: #262730;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                '>
                    <p><strong>Correção {i}:</strong></p>
                    <p>🔍 Erro encontrado: "<span style='color: #ff6b6b'>{sugestao['erro']}</span>"</p>
                    <p>✨ Sugestões: {', '.join(sugestao['sugestoes'][:3])}</p>
                    <p>ℹ️ {sugestao['mensagem']}</p>
                    <p>📍 Contexto: "{sugestao['contexto']}"</p>
                </div>""",
                unsafe_allow_html=True
            )
    
    # Mostrar texto corrigido
    if correcao.texto_corrigido:
        with st.expander("Ver texto com correções aplicadas"):
            st.markdown(
                f"""<div style='
                    background-color: #1a472a;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                '>{correcao.texto_corrigido}</div>""",
                unsafe_allow_html=True
            )

def aplicar_estilos():
    st.markdown("""
        <style>
        /* Reset de cores */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Área de texto principal */
        .stTextArea textarea {
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            line-height: 1.5;
            background-color: #262730;
            color: #FAFAFA !important;
            border: 1px solid #464B5C;
            border-radius: 5px;
            padding: 10px;
        }
        
        /* Box de texto analisado */
        .texto-analise {
            background-color: #FFFFFF;
            color: #000000 !important;
            padding: 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            margin: 10px 0;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Barra de progresso personalizada */
        .stProgress > div > div {
            background-image: linear-gradient(to right, #dc3545, #ffc107, #28a745);
            border-radius: 3px;
            height: 20px;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #262730;
            color: #FAFAFA;
            border-radius: 5px;
            padding: 10px;
            font-weight: bold;
        }
        
        /* Mensagens de erro/sucesso */
        .stAlert {
            background-color: #262730;
            border: 1px solid #464B5C;
            border-radius: 5px;
            padding: 10px;
        }
        
        /* Títulos */
        h1, h2, h3 {
            color: #FAFAFA;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        /* Links */
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        /* Destaque de argumentos e conectivos */
        .argumento-a1 {
            background-color: rgba(76, 175, 80, 0.2);
            border-left: 3px solid #4CAF50;
            padding: 2px 5px;
        }
        
        .argumento-a2 {
            background-color: rgba(33, 150, 243, 0.2);
            border-left: 3px solid #2196F3;
            padding: 2px 5px;
        }
        
        .conectivo {
            border-bottom: 2px dashed;
            padding: 0 2px;
            cursor: help;
        }
        
        /* Tooltips personalizados */
        [title] {
            position: relative;
            cursor: help;
        }
        
        [title]:hover::after {
            content: attr(title);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 5px 10px;
            background: #333;
            color: white;
            border-radius: 3px;
            font-size: 14px;
            white-space: nowrap;
            z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)

def analisar_paragrafo_tempo_real(texto: str, tipo: str) -> AnaliseParagrafo:
    try:
        inicio = datetime.now()
        
        # Análise básica e IA
        analise_basica = analisar_elementos_basicos(texto, tipo)
        analise_ia = None
        palavras = len(texto.split())
        if MIN_PALAVRAS_IA <= palavras <= MAX_PALAVRAS_IA:
            try:
                future = thread_pool.submit(analisar_com_ia, texto, tipo)
                analise_ia = future.result(timeout=API_TIMEOUT)
            except Exception as e:
                logger.warning(f"Falha na análise IA: {e}")
        
        analise_final = combinar_analises(analise_basica, analise_ia)
        feedback = gerar_feedback_completo(analise_final, tipo, texto)
        
        # Análise de argumentos
        analisador_args = AnalisadorArgumentos()
        argumentos = analisador_args.identificar_argumentos(texto, tipo)
        
        # Análise de conectivos
        analisador_conectivos = AnalisadorConectivos()
        analise_conectivos = analisador_conectivos.identificar_conectivos(texto)
        
        # Verificação gramatical
        verificador = VerificadorGramatical()
        correcao_gramatical = verificador.verificar_texto(texto)
        
        tempo_analise = (datetime.now() - inicio).total_seconds()
        
        return AnaliseParagrafo(
            tipo=tipo,
            texto=texto,
            elementos=analise_final,
            feedback=feedback,
            correcao_gramatical=correcao_gramatical,
            argumentos=argumentos,
            analise_conectivos=analise_conectivos,
            tempo_analise=tempo_analise
        )
        
    except Exception as e:
        logger.error(f"Erro na análise em tempo real: {e}")
        return AnaliseParagrafo(
            tipo=tipo,
            texto=texto,
            elementos=analise_basica,
            feedback=gerar_feedback_basico(analise_basica, tipo),
            correcao_gramatical=None,
            argumentos=[],
            analise_conectivos=None,
            tempo_analise=0.0
        )

def gerar_feedback_basico(analise: AnaliseElementos, tipo: str) -> List[str]:
    """Gera feedback básico quando não é possível realizar análise completa."""
    feedback = []
    
    # Feedback sobre elementos
    if analise.presentes:
        feedback.append(f"✅ Elementos identificados: {', '.join(analise.presentes)}")
    if analise.ausentes:
        feedback.append(f"❌ Elementos ausentes: {', '.join(analise.ausentes)}")
    
    # Feedback simplificado baseado no score
    if analise.score >= 0.8:
        feedback.append("🌟 Bom desenvolvimento do parágrafo!")
    elif analise.score >= 0.5:
        feedback.append("📝 Desenvolvimento adequado, mas pode melhorar.")
    else:
        feedback.append("⚠️ Necessário desenvolver melhor o parágrafo.")
    
    return feedback

def gerar_feedback_completo(analise: AnaliseElementos, tipo: str, texto: str) -> List[str]:
    """Gera feedback detalhado combinando análise estrutural."""
    feedback = []
    
    # Identifica elementos bem utilizados
    if analise.presentes:
        elementos_presentes = [e.replace("_", " ").title() for e in analise.presentes]
        feedback.append(
            f"✅ Elementos bem desenvolvidos: {', '.join(elementos_presentes)}"
        )
    
    # Identifica elementos que precisam melhorar
    if analise.ausentes:
        elementos_ausentes = [e.replace("_", " ").title() for e in analise.ausentes]
        feedback.append(
            f"❌ Elementos a melhorar: {', '.join(elementos_ausentes)}"
        )
    
    # Mensagens específicas por tipo de parágrafo
    if tipo == "introducao":
        if analise.score >= 0.6:
            feedback.append("✨ Boa contextualização do tema e apresentação do problema.")
        else:
            feedback.append("💡 Procure contextualizar melhor o tema e apresentar claramente sua tese.")
    elif "desenvolvimento" in tipo:
        if analise.score >= 0.6:
            feedback.append("✨ Argumentação bem estruturada com bom uso de exemplos.")
        else:
            feedback.append("💡 Fortaleça seus argumentos com mais exemplos e explicações.")
    elif tipo == "conclusao":
        if analise.score >= 0.6:
            feedback.append("✨ Proposta de intervenção bem elaborada.")
        else:
            feedback.append("💡 Desenvolva melhor sua proposta de intervenção com agentes e ações claras.")
    
    # Adiciona sugestões específicas
    for sugestao in analise.sugestoes:
        feedback.append(f"💡 {sugestao}")
    
    return feedback

# Markers para análise estrutural
MARKERS = {
    "introducao": {
        "contexto": [
            "atualmente", "nos dias de hoje", "na sociedade contemporânea",
            "no cenário atual", "no contexto", "diante", "perante",
            "em meio a", "frente a", "segundo"
        ],
        "tese": [
            "portanto", "assim", "dessa forma", "logo", "evidencia-se",
            "torna-se", "é fundamental", "é necessário", "é preciso",
            "deve-se considerar", "é importante destacar"
        ],
        "argumentos": [
            "primeiro", "inicialmente", "primeiramente", "além disso",
            "ademais", "outrossim", "não obstante", "por um lado",
            "em primeiro lugar", "sobretudo"
        ]
    },
    "desenvolvimento": {
        "argumento": [
            "com efeito", "de fato", "certamente", "evidentemente",
            "naturalmente", "notadamente", "sobretudo", "principalmente",
            "especialmente", "particularmente"
        ],
        "justificativa": [
            "uma vez que", "visto que", "já que", "pois", "porque",
            "posto que", "considerando que", "tendo em vista que",
            "em virtude de", "devido a"
        ],
        "repertorio": [
            "segundo", "conforme", "de acordo com", "como afirma",
            "como aponta", "como evidencia", "como mostra",
            "segundo dados", "pesquisas indicam", "estudos mostram"
        ],
        "conclusao": [
            "portanto", "assim", "dessa forma", "logo", "por conseguinte",
            "consequentemente", "destarte", "sendo assim",
            "desse modo", "diante disso"
        ]
    },
    "conclusao": {
        "agente": [
            "governo", "estado", "ministério", "secretaria", "município",
            "instituições", "organizações", "sociedade civil",
            "poder público", "autoridades"
        ],
        "acao": [
            "criar", "implementar", "desenvolver", "promover", "estabelecer",
            "formar", "construir", "realizar", "elaborar", "instituir",
            "fomentar", "incentivar"
        ],
        "modo": [
            "por meio de", "através de", "mediante", "por intermédio de",
            "com base em", "utilizando", "a partir de", "por meio da",
            "com o auxílio de", "valendo-se de"
        ],
        "finalidade": [
            "a fim de", "para que", "com o objetivo de", "visando",
            "com a finalidade de", "de modo a", "no intuito de",
            "objetivando", "com o propósito de", "almejando"
        ]
    }
}

def analisar_elementos_basicos(texto: str, tipo: str) -> AnaliseElementos:
    """Realiza análise básica dos elementos do texto."""
    try:
        texto_lower = texto.lower()
        elementos_presentes = []
        elementos_ausentes = []
        
        # Remove números do tipo para mapear corretamente
        tipo_base = tipo.replace("1", "").replace("2", "")
        
        # Verifica presença de markers
        markers = MARKERS[tipo_base]
        for elemento, lista_markers in markers.items():
            encontrado = False
            for marker in lista_markers:
                if marker in texto_lower:
                    elementos_presentes.append(elemento)
                    encontrado = True
                    break
            if not encontrado:
                elementos_ausentes.append(elemento)
        
        # Calcula score baseado na presença de elementos
        total_elementos = len(markers)
        elementos_encontrados = len(elementos_presentes)
        score = elementos_encontrados / total_elementos if total_elementos > 0 else 0.0
        
        # Gera sugestões para elementos ausentes
        sugestoes = []
        for elemento in elementos_ausentes:
            if elemento in SUGESTOES_RAPIDAS:
                sugestoes.append(SUGESTOES_RAPIDAS[elemento][0])
        
        return AnaliseElementos(
            presentes=elementos_presentes,
            ausentes=elementos_ausentes,
            score=score,
            sugestoes=sugestoes
        )
        
    except Exception as e:
        logger.error(f"Erro na análise básica: {e}")
        return AnaliseElementos(
            presentes=[],
            ausentes=[],
            score=0.0,
            sugestoes=["Não foi possível analisar o texto. Tente novamente."]
        )

# Sugestões rápidas para cada elemento
SUGESTOES_RAPIDAS = {
    "contexto": [
        "Desenvolva melhor o contexto histórico ou social do tema",
        "Relacione o tema com a atualidade de forma mais específica",
        "Apresente dados ou informações que contextualizem o tema"
    ],
    "tese": [
        "Apresente seu ponto de vista de forma mais clara e direta",
        "Defina melhor sua posição sobre o tema",
        "Explicite sua opinião sobre a problemática apresentada"
    ],
    "argumentos": [
        "Fortaleça seus argumentos com exemplos concretos",
        "Desenvolva melhor a fundamentação dos argumentos",
        "Apresente evidências que suportem seu ponto de vista"
    ],
}

def analisar_com_ia(texto: str, tipo: str, retry_count: int = 0) -> Optional[AnaliseElementos]:
    """Realiza análise usando IA com tratamento robusto de erros."""
    if retry_count >= MAX_RETRIES or not client:
        return None
        
    try:
        prompt = f"""Analise este {tipo} de redação ENEM e retorne um JSON válido seguindo exatamente este formato, sem adicionar nada mais:

{{
    "elementos_presentes": ["elemento1", "elemento2"],
    "elementos_ausentes": ["elemento3", "elemento4"],
    "sugestoes": ["sugestão 1", "sugestão 2"]
}}

Texto para análise:
{texto}"""

        response = client.chat.completions.create(
            model=ModeloAnalise.RAPIDO,
            messages=[{
                "role": "system",
                "content": "Você é um analisador de redações que retorna apenas JSON válido sem nenhum texto adicional."
            },
            {
                "role": "user",
                "content": prompt
            }],
            temperature=0.3,
            max_tokens=500,
            timeout=API_TIMEOUT,
            response_format={"type": "json_object"}
        )
        
        try:
            resposta_texto = response.choices[0].message.content.strip()
            resposta_texto = resposta_texto[resposta_texto.find("{"):resposta_texto.rfind("}")+1]
            
            resultado = json.loads(resposta_texto)
            
            campos_obrigatorios = ["elementos_presentes", "elementos_ausentes", "sugestoes"]
            if not all(campo in resultado for campo in campos_obrigatorios):
                raise ValueError("Resposta JSON incompleta")
                
            return AnaliseElementos(
                presentes=resultado["elementos_presentes"][:3],
                ausentes=resultado["elementos_ausentes"][:3],
                score=len(resultado["elementos_presentes"]) / (
                    len(resultado["elementos_presentes"]) + len(resultado["elementos_ausentes"]) or 1
                ),
                sugestoes=resultado["sugestoes"][:2]
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Erro no parsing da resposta IA: {e}. Tentativa {retry_count + 1}")
            if retry_count < MAX_RETRIES:
                from time import sleep
                sleep(1)
                return analisar_com_ia(texto, tipo, retry_count + 1)
            return None
            
    except Exception as e:
        logger.error(f"Erro na análise IA: {str(e)}")
        return None

def combinar_analises(analise_basica: AnaliseElementos, analise_ia: Optional[AnaliseElementos]) -> AnaliseElementos:
    """Combina os resultados das análises básica e IA de forma ponderada."""
    if not analise_ia:
        return analise_basica
    
    try:
        # Combina elementos presentes (união dos conjuntos)
        elementos_presentes = list(set(analise_basica.presentes + analise_ia.presentes))
        
        # Combina elementos ausentes (interseção dos conjuntos)
        elementos_ausentes = list(
            set(analise_basica.ausentes).intersection(set(analise_ia.ausentes))
        )
        
        # Média ponderada dos scores (60% básica, 40% IA)
        score = (analise_basica.score * 0.6) + (analise_ia.score * 0.4)
        
        # Combina e prioriza sugestões
        sugestoes_combinadas = []
        
        # Primeiro, adiciona sugestões que aparecem em ambas as análises
        sugestoes_comuns = set(analise_basica.sugestoes).intersection(set(analise_ia.sugestoes))
        sugestoes_combinadas.extend(list(sugestoes_comuns))
        
        # Depois, completa com sugestões únicas até o limite
        sugestoes_restantes = set(analise_basica.sugestoes + analise_ia.sugestoes) - sugestoes_comuns
        sugestoes_combinadas.extend(list(sugestoes_restantes)[:3 - len(sugestoes_combinadas)])
        
        return AnaliseElementos(
            presentes=elementos_presentes,
            ausentes=elementos_ausentes,
            score=min(1.0, max(0.0, score)),  # Garante score entre 0 e 1
            sugestoes=sugestoes_combinadas
        )
        
    except Exception as e:
        logger.error(f"Erro ao combinar análises: {e}")
        return analise_basica

def mostrar_analise_tempo_real(analise: AnaliseParagrafo):
    """Exibe a análise completa em tempo real com todas as análises integradas."""
    st.markdown(f"## Análise do {analise.tipo.title()}")
    
    # Layout principal
    col_texto, col_analise = st.columns([0.6, 0.4])
    
    # Coluna do texto
    with col_texto:
        st.markdown("### 📝 Texto Analisado")
        
        # Aplica todas as marcações no texto
        texto_final = analise.texto
        
        # 1. Destaca conectivos
        if analise.analise_conectivos:
            texto_final = destacar_conectivos(texto_final, analise.analise_conectivos)
        
        # 2. Destaca argumentos
        if analise.argumentos:
            texto_final = destacar_argumentos(texto_final, analise.argumentos)
        
        # 3. Marca erros gramaticais
        if analise.correcao_gramatical and analise.correcao_gramatical.sugestoes:
            texto_final = marcar_erros_no_texto(texto_final, analise.correcao_gramatical)
        
        # Exibe texto com todas as marcações
        st.markdown(
            f"""<div class='texto-analise'>{texto_final}</div>""",
            unsafe_allow_html=True
        )
        
        # Mostra legenda
        mostrar_legenda_conectivos()
    
    # Coluna de análise
    with col_analise:
        # Tabs para diferentes análises
        tab_estrutura, tab_conectivos, tab_gramatical = st.tabs([
            "Estrutura", 
            "Conectivos", 
            "Gramática"
        ])
        
        # Tab de estrutura
        with tab_estrutura:
            if analise.argumentos:
                mostrar_analise_argumentos(analise.argumentos)
            
            st.markdown("#### 📊 Elementos Estruturais")
            col1, col2 = st.columns(2)
            with col1:
                elementos_presentes = len(analise.elementos.presentes)
                st.metric("Elementos Presentes", elementos_presentes)
            with col2:
                st.metric(
                    "Qualidade Estrutural",
                    f"{int(analise.elementos.score * 100)}%"
                )
        
        # Tab de conectivos
        with tab_conectivos:
            if analise.analise_conectivos:
                mostrar_analise_conectivos(analise.analise_conectivos)
            else:
                st.info("Nenhum conectivo identificado.")
        
        # Tab gramatical
        with tab_gramatical:
            if analise.correcao_gramatical:
                mostrar_correcoes_gramaticais(analise.correcao_gramatical)
            else:
                st.info("Análise gramatical não disponível.")
    
    # Feedback geral
    st.markdown("### 💡 Feedback Geral")
    for fb in analise.feedback:
        st.markdown(
            f"""<div style='
                background-color: #1e2a3a;
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
            '>{fb}</div>""",
            unsafe_allow_html=True
        )

def main():
    try:
        # Configurações iniciais
        aplicar_estilos()
        
        # Sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Configurações")
            if 'tema' not in st.session_state:
                st.session_state.tema = "Os desafios relacionados à Cultura do cancelamento na internet"
            
            tema = st.text_area(
                "Tema da Redação",
                value=st.session_state.tema,
                help="Digite o tema para análise mais precisa",
                height=100
            )
            if tema != st.session_state.tema:
                st.session_state.tema = tema
        
        # Interface principal
        st.title("📝 Editor Interativo de Redação ENEM")
        st.markdown("""
            Este editor analisa sua redação em tempo real, fornecendo feedback 
            detalhado para cada parágrafo, incluindo:
            - Análise de argumentos (A1 e A2)
            - Identificação e classificação de conectivos
            - Correção gramatical
            - Sugestões de melhoria estrutural
            
            **Como usar:**
            1. Digite seu texto no editor abaixo
            2. Separe os parágrafos com uma linha em branco
            3. Receba feedback instantâneo sobre cada parágrafo
        """)
        
        # Editor
        texto = st.text_area(
            "Digite sua redação aqui:",
            height=300,
            key="editor_redacao",
            help="Digite ou cole seu texto. Separe os parágrafos com uma linha em branco."
        )
        
        if texto:
            with st.spinner("📊 Analisando sua redação..."):
                paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
                
                if paragrafos:
                    # Tabs para cada parágrafo
                    tabs = st.tabs([
                        f"📄 {detectar_tipo_paragrafo(p, i).title()}" 
                        for i, p in enumerate(paragrafos)
                    ])
                    
                    # Análise em cada tab
                    for i, (tab, paragrafo) in enumerate(zip(tabs, paragrafos)):
                        with tab:
                            tipo = detectar_tipo_paragrafo(paragrafo, i)
                            
                            # Identificador visual do tipo de parágrafo
                            icones = {
                                "introducao": "🎯",
                                "desenvolvimento1": "💡",
                                "desenvolvimento2": "📚",
                                "conclusao": "✨"
                            }
                            st.markdown(f"### {icones.get(tipo, '📝')} {tipo.title()}")
                            
                            # Linha divisória
                            st.markdown("""<hr style="border: 1px solid #464B5C;">""", 
                                      unsafe_allow_html=True)
                            
                            # Análise do parágrafo
                            analise = analisar_paragrafo_tempo_real(paragrafo, tipo)
                            mostrar_analise_tempo_real(analise)
                    
                    # Resumo geral
                    st.markdown("---")
                    st.markdown("### 📊 Visão Geral da Redação")
                    
                    # Métricas gerais
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_paragrafos = len(paragrafos)
                        progresso = min(total_paragrafos / 4, 1.0)
                        st.metric(
                            "Progresso da Redação",
                            f"{int(progresso * 100)}%",
                            f"{total_paragrafos}/4 parágrafos"
                        )
                    
                    with col2:
                        total_palavras = sum(len(p.split()) for p in paragrafos)
                        st.metric(
                            "Total de Palavras",
                            total_palavras,
                            "Meta: 2500-3000"
                        )
                    
                    with col3:
                        if total_paragrafos < 4:
                            proximo = "Conclusão" if total_paragrafos == 3 else f"Desenvolvimento {total_paragrafos + 1}"
                            st.info(f"Próximo: {proximo}")
                        else:
                            st.success("✅ Estrutura Completa!")
                    
                    # Mini mapa dos parágrafos
                    st.markdown("#### 🗺️ Estrutura da Redação")
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        with col:
                            if i < total_paragrafos:
                                st.markdown(
                                    f"""<div style='
                                        background-color: #1a472a;
                                        padding: 10px;
                                        border-radius: 5px;
                                        text-align: center;
                                    '>
                                        {icones.get(detectar_tipo_paragrafo("", i), "📝")}
                                        <br>
                                        {detectar_tipo_paragrafo("", i).title()}
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    """<div style='
                                        background-color: #262730;
                                        padding: 10px;
                                        border-radius: 5px;
                                        text-align: center;
                                        opacity: 0.5;
                                    '>
                                        ➕
                                        <br>
                                        Pendente
                                    </div>""",
                                    unsafe_allow_html=True
                                )
        
        # Footer
        st.markdown("---")
        st.markdown(
            """<div style='text-align: center; opacity: 0.7;'>
            Desenvolvido para auxiliar estudantes na preparação para o ENEM.
            Para feedback e sugestões, use o botão de feedback abaixo de cada análise.
            </div>""",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        st.error(
            "Ocorreu um erro inesperado. Por favor, tente novamente ou entre em contato com o suporte."
        )

if __name__ == "__main__":
    main()



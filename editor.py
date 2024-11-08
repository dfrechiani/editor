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

# Classes de dados
@dataclass
class AnaliseElementos:
    presentes: List[str]
    ausentes: List[str]
    score: float
    sugestoes: List[str]

@dataclass
class AnaliseParagrafo:
    tipo: str
    texto: str
    elementos: AnaliseElementos
    feedback: List[str]
    tempo_analise: float = 0.0

# Configuração da API e modelos
OPENAI_API_KEY = "sk-hhmUJsRrcVymoRI_kfRkq3lQxY5f2VtJtkoisdiwPfT3BlbkFJqHj8CkDnLBsBl14mVWpBHX2VK9yjsBjrLw15TTY8AA"  # Substitua pela sua chave real
client = OpenAI(api_key=OPENAI_API_KEY)

class ModeloAnalise(str, Enum):
    RAPIDO = "gpt-3.5-turbo-1106"

# Configurações de processamento
MAX_WORKERS = 3
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Configurações de timeout e retry
API_TIMEOUT = 10.0  # Aumentado para maior confiabilidade
MAX_RETRIES = 2
MIN_PALAVRAS_IA = 20  # Ajustado para melhor eficiência
MAX_PALAVRAS_IA = 300  # Limite máximo de palavras para análise IA

# Cache com TTL para melhor performance
class CacheAnalise:
    def __init__(self, max_size: int = 50, ttl_seconds: int = 180):
        self.cache: Dict[str, Tuple[AnaliseElementos, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, texto: str, tipo: str) -> Optional[AnaliseElementos]:
        """Recupera análise do cache se existir e não estiver expirada"""
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
        """Armazena análise no cache com gestão de tamanho e expiração"""
        try:
            if len(self.cache) >= self.max_size:
                # Remove itens expirados
                agora = datetime.now()
                expirados = [
                    k for k, (_, t) in self.cache.items() 
                    if (agora - t).seconds >= self.ttl_seconds
                ]
                for k in expirados:
                    del self.cache[k]
                
                # Se ainda estiver cheio, remove o mais antigo
                if len(self.cache) >= self.max_size:
                    mais_antigo = min(self.cache.items(), key=lambda x: x[1][1])
                    del self.cache[mais_antigo[0]]
            
            self.cache[f"{tipo}:{hash(texto)}"] = (analise, datetime.now())
        except Exception as e:
            logger.error(f"Erro ao definir cache: {e}")

# Instância global do cache
cache = CacheAnalise()

# Markers para análise estrutural (com comentários explicativos)
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

# Conectivos para análise de coesão aprimorada
CONECTIVOS = {
    "aditivos": [
        "além disso", "ademais", "também", "e", "outrossim",
        "inclusive", "ainda", "não só... mas também"
    ],
    "adversativos": [
        "mas", "porém", "contudo", "entretanto", "no entanto",
        "todavia", "não obstante", "apesar de", "embora"
    ],
    "conclusivos": [
        "portanto", "logo", "assim", "dessa forma", "por isso",
        "consequentemente", "por conseguinte", "então", "diante disso"
    ],
    "explicativos": [
        "pois", "porque", "já que", "visto que", "uma vez que",
        "posto que", "tendo em vista que", "considerando que"
    ],
    "sequenciais": [
        "primeiramente", "em seguida", "por fim", "depois",
        "anteriormente", "posteriormente", "finalmente"
    ]
}

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
    "argumento": [
        "Apresente evidências para sustentar este argumento",
        "Desenvolva melhor a linha de raciocínio",
        "Utilize dados ou exemplos para fortalecer seu argumento"
    ],
    "justificativa": [
        "Explique melhor o porquê de sua afirmação",
        "Apresente as razões que fundamentam seu argumento",
        "Desenvolva a relação causa-consequência de sua argumentação"
    ],
    "repertorio": [
        "Utilize conhecimentos de outras áreas para enriquecer o texto",
        "Cite exemplos históricos, literários ou científicos",
        "Faça referências a obras, autores ou eventos relevantes"
    ],
    "conclusao": [
        "Relacione melhor a conclusão com os argumentos apresentados",
        "Reforce a solução proposta de forma mais clara",
        "Sintetize os principais pontos discutidos no texto"
    ],
    "agente": [
        "Especifique melhor quem deve executar as ações propostas",
        "Identifique os responsáveis pela implementação da solução",
        "Defina claramente os atores envolvidos na resolução"
    ],
    "acao": [
        "Detalhe melhor as ações necessárias",
        "Especifique as medidas práticas a serem tomadas",
        "Proponha soluções mais concretas e viáveis"
    ],
    "modo": [
        "Explique melhor como as ações devem ser implementadas",
        "Detalhe os meios para alcançar a solução",
        "Especifique os métodos de execução das propostas"
    ],
    "finalidade": [
        "Esclareça melhor os objetivos das ações propostas",
        "Explique qual resultado se espera alcançar",
        "Detalhe as metas e benefícios esperados"
    ]
}

def detectar_tipo_paragrafo(texto: str, posicao: Optional[int] = None) -> str:
    """
    Detecta o tipo do parágrafo baseado em sua posição e conteúdo.
    Prioriza a posição se fornecida, caso contrário analisa o conteúdo.
    """
    try:
        # Detecção por posição (mais confiável)
        if posicao is not None:
            if posicao == 0:
                return "introducao"
            elif posicao in [1, 2]:
                return f"desenvolvimento{posicao}"
            elif posicao == 3:
                return "conclusao"
        
        # Detecção por conteúdo
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
        return "desenvolvimento1"  # Tipo padrão em caso de erro

def analisar_elementos_basicos(texto: str, tipo: str) -> AnaliseElementos:
    """
    Realiza análise básica dos elementos do texto, identificando presença de markers
    e gerando sugestões iniciais.
    """
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
                sugestoes.append(SUGESTOES_RAPIDAS[elemento][0])  # Pega primeira sugestão
        
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

def analisar_com_ia(texto: str, tipo: str, retry_count: int = 0) -> Optional[AnaliseElementos]:
    """Realiza análise usando IA."""
    if retry_count >= MAX_RETRIES or not client:
        return None
        
    try:
        prompt = f"""Analise este {tipo} de redação ENEM:
{texto}

Retorne apenas um objeto JSON com três arrays:
- elementos_presentes: lista de elementos presentes
- elementos_ausentes: lista de elementos ausentes
- sugestoes: lista de sugestões de melhoria

Formato exato: {{"elementos_presentes":[],"elementos_ausentes":[],"sugestoes":[]}}"""

        response = client.chat.completions.create(
            model=ModeloAnalise.RAPIDO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            timeout=API_TIMEOUT,
            response_format={"type": "json_object"}
        )
        
        try:
            resultado = json.loads(response.choices[0].message.content)
            return AnaliseElementos(
                presentes=resultado["elementos_presentes"][:3],
                ausentes=resultado["elementos_ausentes"][:3],
                score=len(resultado["elementos_presentes"]) / (
                    len(resultado["elementos_presentes"]) + len(resultado["elementos_ausentes"]) or 1
                ),
                sugestoes=resultado["sugestoes"][:2]
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Erro no parsing da resposta IA: {e}. Tentativa {retry_count + 1}")
            if retry_count < MAX_RETRIES:
                return analisar_com_ia(texto, tipo, retry_count + 1)
            return None
            
    except Exception as e:
        logger.error(f"Erro na análise IA: {str(e)}")
        return None

def combinar_analises(
    analise_basica: AnaliseElementos,
    analise_ia: Optional[AnaliseElementos]
) -> AnaliseElementos:
    """
    Combina os resultados das análises básica e IA de forma ponderada.
    Prioriza elementos presentes em ambas as análises.
    """
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
        return analise_basica  # Em caso de erro, retorna apenas a análise básica

def analisar_paragrafo_tempo_real(texto: str, tipo: str) -> AnaliseParagrafo:
    """
    Realiza análise completa do parágrafo em tempo real, combinando análise básica e IA.
    """
    try:
        inicio = datetime.now()
        
        # Verifica cache primeiro
        analise_cache = cache.get(texto, tipo)
        if analise_cache:
            return AnaliseParagrafo(
                tipo=tipo,
                texto=texto,
                elementos=analise_cache,
                feedback=gerar_feedback_completo(analise_cache, tipo, texto),
                tempo_analise=(datetime.now() - inicio).total_seconds()
            )
        
        # Análise básica sempre
        analise_basica = analisar_elementos_basicos(texto, tipo)
        
        # Análise IA apenas se texto tiver tamanho adequado
        analise_ia = None
        palavras = len(texto.split())
        if MIN_PALAVRAS_IA <= palavras <= MAX_PALAVRAS_IA:
            try:
                future = thread_pool.submit(analisar_com_ia, texto, tipo)
                analise_ia = future.result(timeout=API_TIMEOUT)
            except Exception as e:
                logger.warning(f"Falha na análise IA: {e}")
                analise_ia = None
        
        # Combina resultados
        analise_final = combinar_analises(analise_basica, analise_ia)
        
        # Gera feedback completo
        feedback = gerar_feedback_completo(analise_final, tipo, texto)
        
        tempo_analise = (datetime.now() - inicio).total_seconds()
        
        resultado = AnaliseParagrafo(
            tipo=tipo,
            texto=texto,
            elementos=analise_final,
            feedback=feedback,
            tempo_analise=tempo_analise
        )
        
        # Cache apenas se análise for rápida e bem-sucedida
        if tempo_analise < 2.0 and feedback:
            cache.set(texto, tipo, analise_final)
        
        return resultado
        
    except Exception as e:
        logger.error(f"Erro na análise em tempo real: {e}")
        # Retorna análise básica em caso de erro
        analise_basica = analisar_elementos_basicos(texto, tipo)
        return AnaliseParagrafo(
            tipo=tipo,
            texto=texto,
            elementos=analise_basica,
            feedback=gerar_feedback_basico(analise_basica, tipo),
            tempo_analise=0.0
        )

def gerar_feedback_completo(analise: AnaliseElementos, tipo: str, texto: str) -> List[str]:
    """
    Gera feedback detalhado combinando análise estrutural e de coesão.
    """
    try:
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
        
        # Análise de coesão
        conectivos_encontrados = []
        for tipo_conectivo, lista_conectivos in CONECTIVOS.items():
            for conectivo in lista_conectivos:
                if conectivo in texto.lower():
                    conectivos_encontrados.append(tipo_conectivo)
                    break
        
        if conectivos_encontrados:
            feedback.append(
                f"📊 Boa utilização de conectivos do tipo: {', '.join(set(conectivos_encontrados))}"
            )
        else:
            feedback.append(
                "💭 Sugestão: Utilize conectivos para melhorar a coesão do texto"
            )
        
        # Adiciona sugestões específicas
        for sugestao in analise.sugestoes:
            feedback.append(f"💡 {sugestao}")
        
        return feedback
        
    except Exception as e:
        logger.error(f"Erro ao gerar feedback: {e}")
        return ["⚠️ Não foi possível gerar o feedback completo. Por favor, tente novamente."]

def gerar_feedback_basico(analise: AnaliseElementos, tipo: str) -> List[str]:
    """
    Gera feedback básico quando não é possível realizar análise completa.
    """
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

def gerar_feedback_score(score: float, tipo: str) -> str:
    """
    Gera feedback específico baseado no score e tipo do parágrafo.
    """
    if score >= 0.8:
        return {
            "introducao": "🌟 Excelente introdução com contextualização e tese claras!",
            "desenvolvimento1": "🌟 Primeiro argumento muito bem desenvolvido!",
            "desenvolvimento2": "🌟 Segundo argumento muito bem estruturado!",
            "conclusao": "🌟 Conclusão muito bem elaborada com proposta clara!"
        }.get(tipo, "🌟 Parágrafo muito bem estruturado!")
    elif score >= 0.5:
        return {
            "introducao": "📝 Introdução adequada, mas pode melhorar a contextualização.",
            "desenvolvimento1": "📝 Primeiro argumento adequado, pode ser fortalecido.",
            "desenvolvimento2": "📝 Segundo argumento adequado, pode ser aprofundado.",
            "conclusao": "📝 Conclusão adequada, pode detalhar melhor as propostas."
        }.get(tipo, "📝 Parágrafo adequado, mas pode melhorar.")
    else:
        return {
            "introducao": "⚠️ Introdução precisa de mais elementos básicos.",
            "desenvolvimento1": "⚠️ Primeiro argumento precisa ser melhor desenvolvido.",
            "desenvolvimento2": "⚠️ Segundo argumento precisa de mais fundamentação.",
            "conclusao": "⚠️ Conclusão precisa de propostas mais concretas."
        }.get(tipo, "⚠️ Parágrafo precisa de mais desenvolvimento.")

def analisar_coesao(texto: str) -> List[str]:
    """
    Analisa a utilização de conectivos e coesão no texto.
    """
    texto_lower = texto.lower()
    conectivos_encontrados = []
    feedback_coesao = []
    
    # Analisa presença de cada tipo de conectivo
    for tipo_conectivo, lista_conectivos in CONECTIVOS.items():
        for conectivo in lista_conectivos:
            if conectivo in texto_lower:
                conectivos_encontrados.append(tipo_conectivo)
                break
    
    # Gera feedback sobre conectivos
    if conectivos_encontrados:
        tipos_conectivos = list(set(conectivos_encontrados))  # Remove duplicatas
        feedback_coesao.append(
            f"📊 Boa utilização de conectivos: {', '.join(tipos_conectivos)}"
        )
    else:
        feedback_coesao.append(
            "💭 Sugestão: Utilize mais conectivos para melhorar a coesão do texto"
        )
    
    return feedback_coesao

def mostrar_analise_tempo_real(analise: AnaliseParagrafo):
    """
    Exibe a análise em tempo real na interface Streamlit com layout aprimorado.
    """
    with st.expander(f"Análise do {analise.tipo.title()}", expanded=True):
        # Layout em duas colunas
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            # Texto do parágrafo com estilo aprimorado
            st.markdown("**Texto do parágrafo:**")
            st.markdown(
                f"""<div style='
                    background-color: #ffffff;
                    padding: 15px;
                    border-radius: 5px;
                    color: #000000;
                    font-family: Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.5;
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>{analise.texto}</div>""",
                unsafe_allow_html=True
            )
        
        with col2:
            # Elementos identificados com ícones
            st.markdown("**Elementos identificados:**")
            for elemento in analise.elementos.presentes:
                st.success(f"✓ {elemento.title()}")
            for elemento in analise.elementos.ausentes:
                st.error(f"✗ {elemento.title()}")
        
        # Barra de progresso com cores dinâmicas
        score_color = get_score_color(analise.elementos.score)
        st.progress(
            analise.elementos.score,
            text=f"Qualidade: {int(analise.elementos.score * 100)}%"
        )

        # Feedback detalhado
        if analise.feedback:
            st.markdown("### Como melhorar seu texto")
            # Removemos o expander aninhado e mostramos direto
            for feedback in analise.feedback:
                st.markdown(
                    get_feedback_html(feedback),
                    unsafe_allow_html=True
                )
        
        # Tempo de análise
        st.caption(
            f"⏱️ Análise realizada em {analise.tempo_analise:.2f} segundos"
        )

def get_score_color(score: float) -> str:
    """
    Retorna a cor apropriada baseada no score.
    """
    if score >= 0.8:
        return "#28a745"  # Verde
    elif score >= 0.5:
        return "#ffc107"  # Amarelo
    else:
        return "#dc3545"  # Vermelho

def get_feedback_html(feedback: str) -> str:
    """
    Gera HTML estilizado para cada item de feedback.
    """
    # Determina o ícone e cor baseado no tipo de feedback
    if feedback.startswith("✅"):
        bg_color = "#1a472a"
        border_color = "#2ecc71"
    elif feedback.startswith("❌"):
        bg_color = "#4a1919"
        border_color = "#e74c3c"
    elif feedback.startswith("💡"):
        bg_color = "#2c3e50"
        border_color = "#3498db"
    else:
        bg_color = "#2C3D4F"
        border_color = "#95a5a6"

    return f"""
        <div style='
            background-color: {bg_color};
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid {border_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <p style='
                color: #FFFFFF;
                margin: 0;
                font-size: 15px;
                line-height: 1.5;
            '>
                {feedback}
            </p>
        </div>
    """

def aplicar_estilos():
    """
    Aplica estilos CSS globais para a interface.
    """
    st.markdown("""
        <style>
        /* Reset de cores para elementos padrão */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Estilo da área de texto principal */
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
        
        /* Estilo para o box de texto analisado */
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
        
        /* Estilo para expanders */
        .streamlit-expanderHeader {
            background-color: #262730;
            color: #FAFAFA;
            border-radius: 5px;
            padding: 10px;
            font-weight: bold;
        }
        
        /* Estilo para mensagens de erro/sucesso */
        .stAlert {
            background-color: #262730;
            border: 1px solid #464B5C;
            border-radius: 5px;
            padding: 10px;
        }
        
        /* Estilo para títulos */
        h1, h2, h3 {
            color: #FAFAFA;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        /* Estilo para links */
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """
    Função principal do aplicativo.
    """
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
            detalhado para cada parágrafo, com sugestões contextualizadas ao tema.
            
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
                    # Análise individual de cada parágrafo
                    for i, paragrafo in enumerate(paragrafos):
                        tipo = detectar_tipo_paragrafo(paragrafo, i)
                        analise = analisar_paragrafo_tempo_real(paragrafo, tipo)
                        mostrar_analise_tempo_real(analise)
                    
                    # Progresso geral
                    total_paragrafos = len(paragrafos)
                    progresso = min(total_paragrafos / 4, 1.0)
                    
                    st.markdown("### Progresso Geral")
                    st.progress(
                        progresso,
                        text=f"Parágrafos: {total_paragrafos}/4 "
                        f"({'Conclusão' if total_paragrafos >= 4 else f'Desenvolvimento {total_paragrafos}'})"
                    )
                    
                    # Dicas baseadas no progresso
                    if total_paragrafos < 4:
                        proximo_tipo = "Conclusão" if total_paragrafos == 3 else f"Desenvolvimento {total_paragrafos + 1}"
                        st.info(f"📝 Próximo passo: Desenvolva o parágrafo de {proximo_tipo}")
                    else:
                        st.success("✨ Parabéns! Você completou todos os parágrafos!")
        
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




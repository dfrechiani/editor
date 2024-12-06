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

# Classes de dados
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
class AnaliseParagrafo:
    tipo: str
    texto: str
    elementos: AnaliseElementos
    feedback: List[str]
    correcao_gramatical: Optional[CorrecaoGramatical] = None
    tempo_analise: float = 0.0

class VerificadorGramatical:
    def __init__(self):
        try:
            self.tool = language_tool_python.LanguageToolPublicAPI('pt-BR')
            self.initialized = True
        except Exception as e:
            logger.error(f"Erro ao inicializar LanguageTool: {e}")
            self.initialized = False
            
    def verificar_texto(self, texto: str) -> CorrecaoGramatical:
        """
        Realiza verificação gramatical completa do texto.
        """
        if not self.initialized:
            return CorrecaoGramatical(
                texto_original=texto,
                sugestoes=[],
                total_erros=0,
                categorias_erros={},
                texto_corrigido=None
            )
            
        try:
            # Obtém todas as correções sugeridas
            matches = self.tool.check(texto)
            
            # Organiza as sugestões por categoria
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
                
                # Aplica a primeira sugestão para gerar texto corrigido
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
        """
        Obtém o contexto do erro no texto.
        """
        start = max(0, offset - context_size)
        end = min(len(texto), offset + length + context_size)
        
        context = texto[start:end]
        if start > 0:
            context = f"...{context}"
        if end < len(texto):
            context = f"{context}..."
            
        return context


# Configuração da API e modelos
import os

# Configurar a chave API via variável de ambiente
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Inicializar o cliente da forma mais simples possível
try:
    client = OpenAI()
except Exception as e:
    st.error("Erro ao inicializar o cliente OpenAI")
    st.error(str(e))
    client = None

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
    """Realiza análise usando IA com tratamento robusto de erros."""
    if retry_count >= MAX_RETRIES or not client:
        return None
        
    try:
        # Prompt mais estruturado e explícito
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
            # Limpa a resposta antes de fazer o parse
            resposta_texto = response.choices[0].message.content.strip()
            # Remove qualquer texto antes ou depois do JSON
            resposta_texto = resposta_texto[resposta_texto.find("{"):resposta_texto.rfind("}")+1]
            
            resultado = json.loads(resposta_texto)
            
            # Validação dos campos obrigatórios
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
                # Espera um pouco antes de tentar novamente
                from time import sleep
                sleep(1)
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

# Função para mostrar as correções gramaticais na interface
def mostrar_correcoes_gramaticais(correcao: CorrecaoGramatical):
    """
    Exibe as correções gramaticais na interface do Streamlit.
    """
    if not correcao or not correcao.sugestoes:
        st.success("✓ Não foram encontrados erros gramaticais significativos.")
        return
        
    st.markdown("### 📝 Correções Gramaticais")
    
    # Resumo dos erros
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de correções sugeridas", correcao.total_erros)
    with col2:
        # Mostra as categorias mais frequentes
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


def analisar_paragrafo_tempo_real(texto: str, tipo: str) -> AnaliseParagrafo:
    try:
        inicio = datetime.now()
        
        # Análise básica e IA (código existente)
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
        
        # Adicionar verificação gramatical
        verificador = VerificadorGramatical()
        correcao_gramatical = verificador.verificar_texto(texto)
        
        tempo_analise = (datetime.now() - inicio).total_seconds()
        
        return AnaliseParagrafo(
            tipo=tipo,
            texto=texto,
            elementos=analise_final,
            feedback=feedback,
            correcao_gramatical=correcao_gramatical,
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
    Exibe a análise completa em tempo real na interface Streamlit com todos os componentes.
    """
    st.markdown(f"## Análise do {analise.tipo.title()}")
    
    # Layout principal em três colunas
    col_texto, col_elementos, col_metricas = st.columns([0.5, 0.25, 0.25])
    
    # Coluna 1: Texto e Análise Principal
    with col_texto:
        st.markdown("### 📝 Texto Analisado")
        
        def marcar_erros_no_texto(texto: str, correcoes: CorrecaoGramatical) -> str:
            if not correcoes or not correcoes.sugestoes:
                return texto
            
            # Ordenamos as sugestões por posição (offset) em ordem decrescente
            sugestoes_ordenadas = sorted(
                correcoes.sugestoes,
                key=lambda x: x['posicao'],
                reverse=True
            )
            
            texto_marcado = texto
            for sugestao in sugestoes_ordenadas:
                erro = sugestao['erro']
                posicao = sugestao['posicao']
                # Criamos uma span com tooltip mostrando a sugestão
                sugestao_texto = sugestao['sugestoes'][0] if sugestao['sugestoes'] else ''
                marcacao = f'<span style="background-color: rgba(255, 107, 107, 0.3); border-bottom: 2px dashed #ff6b6b; cursor: help;" title="Sugestão: {sugestao_texto}">{erro}</span>'
                texto_marcado = (
                    texto_marcado[:posicao] +
                    marcacao +
                    texto_marcado[posicao + len(erro):]
                )
            
            return texto_marcado

        # Aplicar marcações no texto
        texto_com_marcacoes = marcar_erros_no_texto(
            analise.texto,
            analise.correcao_gramatical
        )
        
        # Exibir texto com marcações
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
            '>{texto_com_marcacoes}</div>""",
            unsafe_allow_html=True
        )
        
        # Adicionar legenda após o texto
        if analise.correcao_gramatical and analise.correcao_gramatical.sugestoes:
            st.markdown(
                """<div style='margin-top: 10px; font-size: 14px; color: #666;'>
                    <span style="background-color: rgba(255, 107, 107, 0.3); 
                               border-bottom: 2px dashed #ff6b6b; 
                               padding: 2px 5px;">
                        Texto marcado
                    </span>
                    = Possível erro gramatical (passe o mouse para ver a sugestão)
                </div>""",
                unsafe_allow_html=True
            )
        
        # Contagem de palavras
        palavras = len(analise.texto.split())
        st.caption(f"Total de palavras: {palavras}")
    
    # Coluna 2: Elementos e Estrutura
    with col_elementos:
        st.markdown("### 🎯 Elementos")
        
        # Elementos presentes
        if analise.elementos.presentes:
            for elemento in analise.elementos.presentes:
                st.success(f"✓ {elemento.title()}")
        
        # Elementos ausentes
        if analise.elementos.ausentes:
            for elemento in analise.elementos.ausentes:
                st.error(f"✗ {elemento.title()}")
    
    # Coluna 3: Métricas e Scores
    with col_metricas:
        st.markdown("### 📊 Métricas")
        
        # Score estrutural
        score_color = get_score_color(analise.elementos.score)
        st.metric(
            "Qualidade Estrutural",
            f"{int(analise.elementos.score * 100)}%",
            delta=None,
            delta_color="normal"
        )
        
        # Score gramatical se disponível
        if analise.correcao_gramatical:
            erros = analise.correcao_gramatical.total_erros
            score_gramatical = max(0, 100 - (erros * 10))  # Cada erro reduz 10%
            st.metric(
                "Qualidade Gramatical",
                f"{score_gramatical}%",
                delta=f"-{erros} erros" if erros > 0 else "Sem erros",
                delta_color="inverse"
            )
    
    # Seção de Feedback
    st.markdown("### 💡 Feedback e Sugestões")
    
    # Tabs para diferentes tipos de feedback
    tab_estrutura, tab_gramatical, tab_dicas = st.tabs([
        "Análise Estrutural", 
        "Correções Gramaticais", 
        "Dicas de Melhoria"
    ])
    
    # Tab 1: Análise Estrutural
    with tab_estrutura:
        if analise.feedback:
            for feedback in analise.feedback:
                st.markdown(
                    get_feedback_html(feedback),
                    unsafe_allow_html=True
                )
        else:
            st.info("Nenhum feedback estrutural disponível.")
    
    # Tab 2: Correções Gramaticais
    with tab_gramatical:
        if analise.correcao_gramatical and analise.correcao_gramatical.sugestoes:
            # Resumo das correções
            col_resumo1, col_resumo2 = st.columns(2)
            with col_resumo1:
                st.metric("Total de Correções", analise.correcao_gramatical.total_erros)
            with col_resumo2:
                categorias = sorted(
                    analise.correcao_gramatical.categorias_erros.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                if categorias:
                    st.markdown("**Principais categorias:**")
                    for categoria, count in categorias[:3]:
                        st.markdown(f"- {categoria}: {count}")
            
            # Lista detalhada de correções
            st.markdown("#### Detalhamento das Correções")
            for i, sugestao in enumerate(analise.correcao_gramatical.sugestoes, 1):
                st.markdown(
                    f"""<div style='
                        background-color: #262730;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 5px 0;
                    '>
                        <p><strong>Correção {i}:</strong></p>
                        <p>🔍 Erro: "<span style='color: #ff6b6b'>{sugestao['erro']}</span>"</p>
                        <p>✨ Sugestão: {', '.join(sugestao['sugestoes'][:1])}</p>
                        <p>ℹ️ {sugestao['mensagem']}</p>
                        <p>📍 Contexto: "{sugestao['contexto']}"</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            # Texto corrigido
            st.markdown("#### Versão Corrigida")
            if analise.correcao_gramatical.texto_corrigido:
                st.markdown(
                    f"""<div style='
                        background-color: #1a472a;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 10px 0;
                    '>{analise.correcao_gramatical.texto_corrigido}</div>""",
                    unsafe_allow_html=True
                )
        else:
            st.success("✓ Não foram encontrados erros gramaticais significativos.")
    
    # Tab 3: Dicas de Melhoria
    with tab_dicas:
        dicas = get_dicas_por_tipo(analise.tipo, analise.elementos.score)
        for dica in dicas:
            st.markdown(
                f"""<div style='
                    background-color: #1e2a3a;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                '>
                    💡 {dica}
                </div>""",
                unsafe_allow_html=True
            )
    
    # Rodapé com metadados
    st.markdown("---")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    
    with col_meta1:
        st.caption(f"⏱️ Tempo de análise: {analise.tempo_analise:.2f}s")
    
    with col_meta2:
        st.caption(f"📊 Modelo: {'IA + Básica' if analise.elementos.score > 0.3 else 'Básica'}")
    
    with col_meta3:
        # Botão de feedback
        if st.button("📝 Reportar Análise", key=f"report_{hash(analise.texto)}"):
            st.info("Feedback registrado. Obrigado pela contribuição!")

def get_dicas_por_tipo(tipo: str, score: float) -> List[str]:
    """
    Retorna dicas específicas baseadas no tipo do parágrafo e score.
    """
    dicas_base = {
        "introducao": [
            "Apresente o tema de forma gradual, partindo do geral para o específico",
            "Inclua uma tese clara e bem definida ao final",
            "Use dados ou fatos relevantes para contextualizar o tema"
        ],
        "desenvolvimento1": [
            "Desenvolva um argumento principal forte logo no início",
            "Use exemplos concretos para sustentar seu ponto de vista",
            "Mantenha o foco na tese apresentada na introdução"
        ],
        "desenvolvimento2": [
            "Apresente um novo aspecto do tema, complementar ao primeiro desenvolvimento",
            "Estabeleça conexões claras com os argumentos anteriores",
            "Utilize repertório sociocultural relevante"
        ],
        "conclusao": [
            "Retome os principais pontos discutidos de forma sintética",
            "Proponha soluções viáveis e bem estruturadas",
            "Especifique agentes, ações e meios para implementação"
        ]
    }
    
    # Adiciona dicas baseadas no score
    dicas = dicas_base.get(tipo, [])
    if score < 0.5:
        dicas.append("⚠️ Reforce a estrutura básica do parágrafo")
    elif score < 0.8:
        dicas.append("📈 Adicione mais elementos de conexão entre as ideias")
    
    return dicas

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


def pagina_analise():
    """
    Exibe a análise detalhada da redação usando as competências do ENEM.
    """
    st.title("Análise da Redação")

    # Verificar se temos os dados necessários
    if 'redacao_texto' not in st.session_state or 'resultados' not in st.session_state:
        st.warning("Por favor, submeta uma redação primeiro para análise.")
        if st.button("Ir para Editor de Redação"):
            st.session_state.page = 'editor'
            st.rerun()
        return

    # Recuperar dados da sessão
    redacao_texto = st.session_state.redacao_texto
    resultados = st.session_state.resultados
    tema_redacao = st.session_state.tema_redacao

    # Layout principal
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Texto da Redação:")
        st.write(f"**Tema:** {tema_redacao}")
        
        # Mostrar redação com erros destacados
        competencia_selecionada = st.selectbox(
            "Visualizar erros por competência:",
            list(competencies.keys()),
            format_func=lambda x: competencies[x]
        )
        
        texto_marcado, _ = marcar_erros_por_competencia(
            redacao_texto,
            resultados['erros_especificos'],
            competencia_selecionada
        )
        st.markdown(texto_marcado, unsafe_allow_html=True)

    with col2:
        st.subheader("Análise por Competência")
        for comp, desc in competencies.items():
            with st.expander(f"{desc} - {resultados['notas'][comp]}/200"):
                erros = resultados['erros_especificos'][comp]
                if erros:
                    st.write("**Erros encontrados:**")
                    for erro in erros:
                        st.markdown(f"""
                        - **Trecho:** {erro['trecho']}
                        - **Explicação:** {erro['explicação']}
                        - **Sugestão:** {erro['sugestão']}
                        ---
                        """)
                else:
                    st.success("Não foram encontrados erros nesta competência!")
                
                st.write("**Análise Detalhada:**")
                st.write(resultados['analises_detalhadas'][comp])

    # Visualização das notas
    st.subheader("Desempenho Geral")
    col1, col2 = st.columns([2,1])
    
    with col1:
        # Gráfico de barras das notas
        criar_grafico_barras(resultados['notas'])
    
    with col2:
        # Nota total e médias
        st.metric("Nota Total", f"{resultados['nota_total']}/1000")
        media = resultados['nota_total']/5
        st.metric("Média por Competência", f"{media:.1f}/200")
    
    # Botões de navegação
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Voltar ao Editor"):
            st.session_state.page = 'editor'
            st.rerun()
    with col2:
        if st.button("Ver Trilhas de Aprendizado →"):
            st.session_state.page = 'trilhas'
            st.rerun()

def criar_grafico_barras(notas: Dict[str, int]):
    """
    Cria um gráfico de barras mostrando as notas por competência.
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(competencies.values()),
            y=[notas[comp] for comp in competencies.keys()],
            marker_color=[competency_colors[comp] for comp in competencies.keys()]
        )
    ])
    
    fig.update_layout(
        title="Notas por Competência",
        xaxis_title="Competência",
        yaxis_title="Pontuação",
        yaxis_range=[0, 200],
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """
    Função principal do aplicativo, controlando a navegação e funcionalidades.
    Inclui gestão de estado, navegação entre páginas, e tratamento de erros.
    """
    try:
        # Inicialização do estado global
        if 'page' not in st.session_state:
            st.session_state.page = 'editor'
        if 'tema' not in st.session_state:
            st.session_state.tema_redacao = "Os desafios relacionados à Cultura do cancelamento na internet"
        if 'redacao_texto' not in st.session_state:
            st.session_state.redacao_texto = ""
        if 'resultados' not in st.session_state:
            st.session_state.resultados = None
        if 'historico_analises' not in st.session_state:
            st.session_state.historico_analises = []

        # Inicialização dos verificadores
        verificador_gramatical = VerificadorGramatical()
            
        # Navegação entre páginas
        if st.session_state.page == 'editor':
            # Configurações iniciais
            aplicar_estilos()
            
            # Sidebar com configurações
            with st.sidebar:
                st.markdown("### ⚙️ Configurações")
                
                # Configuração do tema
                tema = st.text_area(
                    "Tema da Redação",
                    value=st.session_state.tema_redacao,
                    help="Digite o tema para análise mais precisa",
                    height=100
                )
                if tema != st.session_state.tema_redacao:
                    st.session_state.tema_redacao = tema
                
                # Configurações adicionais
                with st.expander("Configurações Avançadas"):
                    st.checkbox(
                        "Ativar análise gramatical em tempo real",
                        value=True,
                        key="analise_gramatical_real_time"
                    )
                    st.checkbox(
                        "Mostrar sugestões detalhadas",
                        value=True,
                        key="mostrar_sugestoes_detalhadas"
                    )
            
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
            
            # Editor principal
            texto = st.text_area(
                "Digite sua redação aqui:",
                height=300,
                key="editor_redacao",
                help="Digite ou cole seu texto. Separe os parágrafos com uma linha em branco.",
                value=st.session_state.redacao_texto
            )
            
            # Atualiza o estado quando o texto muda
            if texto != st.session_state.redacao_texto:
                st.session_state.redacao_texto = texto
                st.session_state.resultados = None  # Reseta resultados anteriores
            
            if texto:
                with st.spinner("📊 Analisando sua redação..."):
                    paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
                    
                    if paragrafos:
                        # Sistema de tabs para análise de parágrafos
                        tabs = st.tabs([
                            f"📄 {detectar_tipo_paragrafo(p, i).title()}" 
                            for i, p in enumerate(paragrafos)
                        ])
                        
                        # Análise em cada tab
                        analises_paragrafos = []
                        for i, (tab, paragrafo) in enumerate(zip(tabs, paragrafos)):
                            with tab:
                                tipo = detectar_tipo_paragrafo(paragrafo, i)
                                
                                # Adiciona identificador visual do tipo de parágrafo
                                icones = {
                                    "introducao": "🎯",
                                    "desenvolvimento1": "💡",
                                    "desenvolvimento2": "📚",
                                    "conclusao": "✨"
                                }
                                st.markdown(f"### {icones.get(tipo, '📝')} {tipo.title()}")
                                
                                # Linha divisória visual
                                st.markdown("""<hr style="border: 1px solid #464B5C;">""", 
                                          unsafe_allow_html=True)
                                
                                # Análise do parágrafo
                                analise = analisar_paragrafo_tempo_real(paragrafo, tipo)
                                mostrar_analise_tempo_real(analise)
                                analises_paragrafos.append(analise)
                        
                        # Armazena análises no histórico
                        st.session_state.historico_analises = analises_paragrafos
                        
                        # Resumo geral após as tabs
                        st.markdown("---")
                        st.markdown("### 📊 Visão Geral da Redação")
                        
                        # Métricas gerais em colunas
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
                            status_palavras = "✅" if 2500 <= total_palavras <= 3000 else "⚠️"
                            st.metric(
                                "Total de Palavras",
                                f"{status_palavras} {total_palavras}",
                                "Meta: 2500-3000"
                            )
                        
                        with col3:
                            # Cálculo da nota estimada
                            media_scores = sum(a.elementos.score for a in analises_paragrafos) / len(analises_paragrafos)
                            nota_estimada = int(media_scores * 1000)
                            st.metric(
                                "Nota Estimada",
                                f"{nota_estimada}/1000",
                                "Baseada na análise atual"
                            )
                        
                        # Botões de ação
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Salvar Rascunho", use_container_width=True):
                                st.session_state.ultimo_rascunho = texto
                                st.success("Rascunho salvo com sucesso!")
                                
                        with col2:
                            if st.button("📊 Análise Completa", type="primary", use_container_width=True):
                                st.session_state.page = 'analise'
                                st.rerun()
            
            # Footer informativo
            st.markdown("---")
            st.markdown(
                """<div style='text-align: center; opacity: 0.7;'>
                Desenvolvido para auxiliar estudantes na preparação para o ENEM.
                Para feedback e sugestões, use o botão de feedback abaixo de cada análise.
                </div>""",
                unsafe_allow_html=True
            )
                
        elif st.session_state.page == 'analise':
            pagina_analise()
            
        elif st.session_state.page == 'trilhas':
            # Implementação futura da página de trilhas de aprendizado
            st.title("🎯 Trilhas de Aprendizado")
            st.info("Esta funcionalidade será implementada em breve!")
            if st.button("← Voltar à Análise"):
                st.session_state.page = 'analise'
                st.rerun()
            
    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        st.error(
            """Ocorreu um erro inesperado. Por favor, tente novamente ou entre em contato com o suporte.
            
            Detalhes técnicos: {str(e)}"""
        )
        
        # Botão de recuperação
        if st.button("🔄 Reiniciar Aplicativo"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

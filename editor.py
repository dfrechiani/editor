import os
import streamlit as st
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import plotly.graph_objects as go
from collections import Counter
import spacy
from anthropic import Anthropic
from elevenlabs import Voice, TextToSpeech, set_api_key


# Configuração inicial do Streamlit
st.set_page_config(
    page_title="Sistema de Redação ENEM",
    page_icon="📝",
    layout="wide"
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização do cliente Anthropic
try:
    anthropic_client = Anthropic(api_key=st.secrets["anthropic"]["api_key"])
    
    # Configurações do modelo SpaCy
    if 'nlp' not in st.session_state:
        st.session_state.nlp = spacy.load('pt_core_news_sm')
    
except Exception as e:
    logger.error(f"Erro na inicialização dos clientes: {e}")
    st.error("Erro ao inicializar conexões. Por favor, tente novamente mais tarde.")

set_api_key(st.secrets["elevenlabs"]["api_key"])

# Constantes
COMPETENCIES = {
    "competency1": "Domínio da Norma Culta",
    "competency2": "Compreensão do Tema",
    "competency3": "Seleção e Organização das Informações",
    "competency4": "Conhecimento dos Mecanismos Linguísticos",
    "competency5": "Proposta de Intervenção"
}

COMPETENCY_COLORS = {
    "competency1": "#FF6B6B",
    "competency2": "#4ECDC4",
    "competency3": "#45B7D1",
    "competency4": "#FFA07A",
    "competency5": "#98D8C8"
}

# Inicialização do estado da sessão
if 'page' not in st.session_state:
    st.session_state.page = 'envio'

def pagina_envio_redacao():
    """Página principal de envio de redação"""
    st.title("Sistema de Análise de Redação ENEM")

    # Barra lateral para navegação
    with st.sidebar:
        if st.button("Nova Redação", key="nova_redacao"):
            st.session_state.page = 'envio'
            st.rerun()
        if st.button("Tutoria", key="tutoria"):
            st.session_state.page = 'tutoria'
            st.rerun()

    # Campo para tema
    tema_redacao = st.text_input("Tema da redação:")

    # Inicializar texto_redacao no session_state se não existir
    if 'texto_redacao' not in st.session_state:
        st.session_state.texto_redacao = ""

    # Campo de texto para digitação
    texto_redacao = st.text_area(
        "Digite sua redação aqui:", 
        value=st.session_state.texto_redacao,
        height=400,
        key="area_redacao"
    )

    # Upload de arquivo txt
    st.write("Ou faça upload de um arquivo .txt")
    uploaded_file = st.file_uploader("", type=['txt'], key="uploader")
    
    # Processar arquivo txt se fornecido
    if uploaded_file is not None and not texto_redacao:
        texto_redacao = uploaded_file.getvalue().decode("utf-8")
        st.session_state.texto_redacao = texto_redacao

    # Botão de processamento
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if tema_redacao:
            if st.button("Analisar Redação", key="processar_redacao", use_container_width=True):
                if texto_redacao:
                    with st.spinner("Analisando redação..."):
                        try:
                            # Análise com cohmetrix
                            cohmetrix_results = analyze_with_cohmetrix(texto_redacao)
                            
                            # Processar redação
                            resultados = processar_redacao_completa(
                                texto_redacao, 
                                tema_redacao, 
                                cohmetrix_results
                            )
                            
                            if resultados:
                                # Atualizar estados da sessão
                                st.session_state.update({
                                    'resultados': resultados,
                                    'tema_redacao': tema_redacao,
                                    'redacao_texto': texto_redacao,
                                    'texto_redacao': "",  # Limpar campo após processamento
                                })
                                
                                st.success("Redação processada com sucesso!")
                                st.session_state.page = 'resultado'
                                st.rerun()
                            else:
                                st.error("Não foi possível processar a redação.")
                        except Exception as e:
                            st.error("Erro ao processar a redação.")
                            logging.error(f"Erro ao processar redação: {str(e)}", exc_info=True)
                else:
                    st.warning("Por favor, insira o texto da redação antes de processar.")
        else:
            st.button("Analisar Redação", key="processar_redacao", 
                     disabled=True, use_container_width=True)
            st.warning("Por favor, forneça o tema da redação antes de processar.")

def pagina_resultado_analise():
    """Página de exibição dos resultados da análise"""
    st.title("Resultado da Análise")

    if 'resultados' not in st.session_state:
        st.warning("Nenhuma análise disponível. Por favor, envie uma redação.")
        if st.button("Voltar para Envio"):
            st.session_state.page = 'envio'
            st.rerun()
        return

    # Dados da análise
    resultados = st.session_state.resultados
    tema_redacao = st.session_state.tema_redacao
    texto_redacao = st.session_state.redacao_texto

    # Mostrar tema
    st.subheader(f"Tema: {tema_redacao}")

    # Criar tabs para cada competência
    tabs = st.tabs([COMPETENCIES[comp] for comp in COMPETENCIES])
    
    for i, (comp, tab) in enumerate(zip(COMPETENCIES, tabs)):
        with tab:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Análise detalhada
                st.markdown("#### Análise Detalhada")
                st.markdown(resultados['analises_detalhadas'][comp])
                
                # Mostrar erros específicos
                if resultados['erros_especificos'].get(comp):
                    st.markdown("#### Erros Identificados")
                    for erro in resultados['erros_especificos'][comp]:
                        with st.expander(f"Erro: {erro['descrição']}"):
                            st.write(f"Trecho: '{erro['trecho']}'")
                            st.write(f"Explicação: {erro['explicação']}")
                            st.write(f"Sugestão: {erro['sugestão']}")
            
            with col2:
                # Nota e justificativa
                st.metric(
                    "Nota",
                    f"{resultados['notas'][comp]}/200",
                    delta=None
                )
                if comp in resultados['justificativas']:
                    st.write("**Justificativa da nota:**")
                    st.write(resultados['justificativas'][comp])

    # Nota total
    st.metric(
        "Nota Total",
        f"{resultados['nota_total']}/1000",
        delta=None
    )

    # Botões de navegação
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Nova Redação"):
            st.session_state.page = 'envio'
            st.rerun()
    with col2:
        if st.button("Iniciar Tutoria"):
            st.session_state.page = 'tutoria'
            st.rerun()

def pagina_tutoria():
    """Página principal do sistema de tutoria inteligente"""
    st.title("Tutoria Personalizada")

    # Verificar se há análise disponível
    if 'resultados' not in st.session_state:
        st.warning("É necessário analisar uma redação primeiro para iniciar a tutoria.")
        if st.button("Enviar Redação"):
            st.session_state.page = 'envio'
            st.rerun()
        return

    # Inicializar estados da tutoria se necessário
    if 'tutoria_estado' not in st.session_state:
        st.session_state.tutoria_estado = {
            'etapa': 'diagnostico',
            'competencia_foco': None,
            'exercicios_completos': set(),
            'pontuacao': 0
        }

    # Sidebar com progresso e informações
    with st.sidebar:
        st.subheader("Seu Progresso")
        st.progress(calcular_progresso_tutoria())
        st.metric("Pontuação", st.session_state.tutoria_estado['pontuacao'])

    # Lógica principal da tutoria baseada na etapa atual
    etapa = st.session_state.tutoria_estado['etapa']

    if etapa == 'diagnostico':
        realizar_diagnostico()
    elif etapa == 'plano_estudo':
        mostrar_plano_estudo()
    elif etapa == 'exercicios':
        realizar_exercicios()
    elif etapa == 'feedback':
        mostrar_feedback_final()

def realizar_diagnostico():
    """Realiza diagnóstico inicial e identifica competência foco"""
    st.subheader("Diagnóstico Inicial")

    # Encontrar competência com menor nota
    notas = st.session_state.resultados['notas']
    competencia_foco = min(notas.items(), key=lambda x: x[1])[0]
    
    # Exibir análise geral
    st.write("Com base na sua última redação, identificamos:")
    
    # Criar gráfico radar das competências
    criar_grafico_radar(notas)
    
    # Mostrar competência foco
    st.info(f"📍 Foco Recomendado: {COMPETENCIES[competencia_foco]}")
    st.write(f"Nota atual: {notas[competencia_foco]}/200")
    
    if st.button("Iniciar Plano de Estudos"):
        st.session_state.tutoria_estado['competencia_foco'] = competencia_foco
        st.session_state.tutoria_estado['etapa'] = 'plano_estudo'
        st.rerun()

def mostrar_plano_estudo():
    """Mostra e gerencia o plano de estudos personalizado"""
    st.subheader("Seu Plano de Estudos")
    
    comp = st.session_state.tutoria_estado['competencia_foco']
    erros = st.session_state.resultados['erros_especificos'].get(comp, [])
    
    # Gerar plano de estudos usando Claude
    if 'plano_estudo' not in st.session_state.tutoria_estado:
        plano = gerar_plano_estudo(comp, erros)
        st.session_state.tutoria_estado['plano_estudo'] = plano
    
    plano = st.session_state.tutoria_estado['plano_estudo']
    
    # Mostrar objetivos
    st.markdown("### 🎯 Objetivos")
    for obj in plano['objetivos']:
        st.write(f"- {obj}")
    
    # Mostrar módulos de estudo
    st.markdown("### 📚 Módulos de Estudo")
    for i, modulo in enumerate(plano['modulos'], 1):
        with st.expander(f"Módulo {i}: {modulo['titulo']}"):
            st.write(modulo['descricao'])
            st.write("**Conceitos-chave:**")
            for conceito in modulo['conceitos']:
                st.write(f"- {conceito}")
    
    if st.button("Começar Exercícios"):
        st.session_state.tutoria_estado['etapa'] = 'exercicios'
        st.rerun()

def realizar_exercicios():
    """Gerencia a realização dos exercícios práticos"""
    st.subheader("Exercícios Práticos")
    
    comp = st.session_state.tutoria_estado['competencia_foco']
    
    # Gerar exercício se necessário
    if 'exercicio_atual' not in st.session_state.tutoria_estado:
        exercicio = gerar_exercicio(comp)
        st.session_state.tutoria_estado['exercicio_atual'] = exercicio
    
    exercicio = st.session_state.tutoria_estado['exercicio_atual']
    
    # Mostrar exercício
    st.markdown(f"### {exercicio['titulo']}")
    st.write(exercicio['instrucoes'])
    
    # Campo para resposta
    resposta = st.text_area("Sua resposta:", height=200)
    
    if st.button("Verificar"):
        feedback = avaliar_resposta(exercicio, resposta, comp)
        st.write(feedback['comentario'])
        
        # Gerar áudio do feedback
        audio = gerar_audio_feedback(feedback['comentario'])
        st.audio(audio)
        
        # Atualizar pontuação
        st.session_state.tutoria_estado['pontuacao'] += feedback['pontos']
        
        # Opção para próximo exercício
        if st.button("Próximo Exercício"):
            del st.session_state.tutoria_estado['exercicio_atual']
            st.rerun()

def mostrar_feedback_final():
    """Mostra feedback final e recomendações"""
    st.subheader("Feedback Final")
    
    comp = st.session_state.tutoria_estado['competencia_foco']
    pontuacao = st.session_state.tutoria_estado['pontuacao']
    
    # Gerar feedback geral
    feedback = gerar_feedback_final(comp, pontuacao)
    
    st.markdown("### 🎉 Parabéns pelo seu progresso!")
    st.write(feedback['mensagem'])
    
    # Mostrar estatísticas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pontuação Total", pontuacao)
    with col2:
        st.metric("Exercícios Completados", len(st.session_state.tutoria_estado['exercicios_completos']))
    
    # Recomendações
    st.markdown("### 📚 Recomendações para Continuar")
    for rec in feedback['recomendacoes']:
        st.write(f"- {rec}")
    
    # Opções de navegação
    if st.button("Nova Redação"):
        st.session_state.page = 'envio'
        st.rerun()

def gerar_plano_estudo(competencia: str, erros: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gera plano de estudos personalizado usando Claude"""
    prompt = f"""
    Com base na seguinte análise de uma redação do ENEM para a competência {COMPETENCIES[competencia]}:
    
    Erros identificados:
    {json.dumps(erros, indent=2)}
    
    Crie um plano de estudos personalizado que inclua:
    1. 3 objetivos claros e específicos
    2. 4 módulos de estudo progressivos
    3. Conceitos-chave para cada módulo
    
    O plano deve seguir uma progressão lógica e abordar os erros identificados.
    
    Responda em formato JSON com a seguinte estrutura:
    {
        "objetivos": ["objetivo 1", "objetivo 2", "objetivo 3"],
        "modulos": [
            {
                "titulo": "título do módulo",
                "descricao": "descrição detalhada",
                "conceitos": ["conceito 1", "conceito 2", "conceito 3"]
            }
        ]
    }
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Erro ao gerar plano de estudos: {e}")
        return {
            "objetivos": ["Melhorar compreensão dos conceitos básicos"],
            "modulos": [
                {
                    "titulo": "Módulo Básico",
                    "descricao": "Revisão dos conceitos fundamentais",
                    "conceitos": ["Conceito básico 1", "Conceito básico 2"]
                }
            ]
        }

def gerar_exercicio(competencia: str) -> Dict[str, Any]:
    """Gera exercício personalizado baseado na competência"""
    prompt = f"""
    Crie um exercício prático para desenvolver habilidades na competência {COMPETENCIES[competencia]} do ENEM.
    
    O exercício deve:
    1. Ser específico e focado
    2. Incluir instruções claras
    3. Permitir prática objetiva
    4. Ter critérios claros de avaliação
    
    Responda em formato JSON com a seguinte estrutura:
    {{
        "titulo": "título do exercício",
        "instrucoes": "instruções detalhadas",
        "criterios": ["critério 1", "critério 2", "critério 3"],
        "exemplo": "exemplo de resposta esperada"
    }}
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Erro ao gerar exercício: {e}")
        return {
            "titulo": "Exercício Básico",
            "instrucoes": "Desenvolva um parágrafo sobre o tema dado",
            "criterios": ["Clareza", "Coesão", "Adequação"],
            "exemplo": "Exemplo de resposta adequada"
        }

def avaliar_resposta(exercicio: Dict[str, Any], resposta: str, competencia: str) -> Dict[str, Any]:
    """Avalia resposta do exercício usando Claude"""
    prompt = f"""
    Avalie a seguinte resposta para um exercício de {COMPETENCIES[competencia]}:
    
    Exercício:
    {exercicio['instrucoes']}
    
    Critérios:
    {json.dumps(exercicio['criterios'], indent=2)}
    
    Resposta do aluno:
    {resposta}
    
    Forneça:
    1. Análise detalhada
    2. Pontos positivos
    3. Pontos de melhoria
    4. Sugestões específicas
    5. Pontuação (0-10)
    
    Responda em formato JSON:
    {{
        "comentario": "feedback detalhado",
        "pontos_positivos": ["ponto 1", "ponto 2"],
        "pontos_melhoria": ["melhoria 1", "melhoria 2"],
        "sugestoes": ["sugestão 1", "sugestão 2"],
        "pontos": int
    }}
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Erro ao avaliar resposta: {e}")
        return {
            "comentario": "Não foi possível avaliar a resposta",
            "pontos_positivos": [],
            "pontos_melhoria": [],
            "sugestoes": [],
            "pontos": 5
        }

from elevenlabs import generate_audio


def gerar_audio_feedback(texto: str) -> bytes:
    """Gera áudio do feedback usando ElevenLabs com voz em português do Brasil"""
    try:
        # Configure a chave da API a partir do Streamlit Secrets
        set_api_key(st.secrets["elevenlabs"]["api_key"])

        # Gere o áudio com o texto fornecido, especificando a voz em português
        audio = generate(
            text=texto,
            voice="Camila",  # Substitua por uma voz brasileira disponível na ElevenLabs
            model="eleven_multilingual_v1"
        )

        return audio  # Retorna o áudio como bytes
    except Exception as e:
        logger.error(f"Erro ao gerar áudio: {e}")
        return b""  # Retorna bytes vazios em caso de erro

def calcular_progresso_tutoria() -> float:
    """Calcula o progresso atual na trilha de tutoria"""
    etapas = {
        'diagnostico': 0.25,
        'plano_estudo': 0.5,
        'exercicios': 0.75,
        'feedback': 1.0
    }
    return etapas.get(st.session_state.tutoria_estado['etapa'], 0)

def gerar_feedback_final(competencia: str, pontuacao: int) -> Dict[str, Any]:
    """Gera feedback final da tutoria"""
    prompt = f"""
    Gere um feedback final para um aluno que completou a tutoria em {COMPETENCIES[competencia]}
    com pontuação {pontuacao}.
    
    Inclua:
    1. Mensagem motivacional
    2. Resumo do progresso
    3. 3 recomendações específicas para continuar o desenvolvimento
    
    Responda em formato JSON:
    {{
        "mensagem": "mensagem personalizada",
        "recomendacoes": ["recomendação 1", "recomendação 2", "recomendação 3"]
    }}
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Erro ao gerar feedback final: {e}")
        return {
            "mensagem": "Parabéns pelo seu progresso!",
            "recomendacoes": [
                "Continue praticando regularmente",
                "Revise os conceitos aprendidos",
                "Aplique as técnicas em novas redações"
            ]
        }

def criar_grafico_radar(notas: Dict[str, int]):
    """Cria gráfico radar das competências"""
    categorias = list(COMPETENCIES.values())
    valores = list(notas.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=valores,
        theta=categorias,
        fill='toself',
        line=dict(color='#4CAF50')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 200]
            )
        ),
        showlegend=False,
        title={
            'text': 'Perfil de Competências',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Função principal que controla o fluxo da aplicação"""
    # Inicialização do estado se necessário
    if 'page' not in st.session_state:
        st.session_state.page = 'envio'
    
    # Barra lateral de navegação
    with st.sidebar:
        st.title("📝 Menu")
        
        if st.button("Nova Redação 📝"):
            st.session_state.page = 'envio'
            st.rerun()
        
        if 'resultados' in st.session_state:
            if st.button("Ver Análise 📊"):
                st.session_state.page = 'resultado'
                st.rerun()
            
            if st.button("Tutoria 👨‍🏫"):
                st.session_state.page = 'tutoria'
                st.rerun()
        
        # Mostrar informações de progresso se estiver na tutoria
        if st.session_state.page == 'tutoria' and 'tutoria_estado' in st.session_state:
            st.divider()
            st.subheader("Progresso")
            st.progress(calcular_progresso_tutoria())
            st.metric("Pontuação", st.session_state.tutoria_estado.get('pontuacao', 0))
    
    # Roteamento de páginas
    if st.session_state.page == 'envio':
        pagina_envio_redacao()
    elif st.session_state.page == 'resultado':
        if 'resultados' in st.session_state:
            pagina_resultado_analise()
        else:
            st.warning("Nenhuma análise disponível. Envie uma redação primeiro.")
            st.session_state.page = 'envio'
            st.rerun()
    elif st.session_state.page == 'tutoria':
        if 'resultados' in st.session_state:
            pagina_tutoria()
        else:
            st.warning("Nenhuma análise disponível. Envie uma redação primeiro.")
            st.session_state.page = 'envio'
            st.rerun()
    else:
        st.error("Página não encontrada")
        st.session_state.page = 'envio'
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")
        logger.error(f"Erro inesperado na aplicação: {str(e)}", exc_info=True)

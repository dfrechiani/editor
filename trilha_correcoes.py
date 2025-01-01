import json
import logging
from typing import Any, Dict, List

# Verifique se o Streamlit está instalado antes de importar
try:
    import streamlit as st
    # Configuração inicial do Streamlit
    st.set_page_config(page_title="Trilha de Competências - ENEM", page_icon="📝", layout="wide")
except ModuleNotFoundError as e:
    raise RuntimeError("O módulo Streamlit não está instalado no ambiente. Certifique-se de que Streamlit esteja disponível antes de executar o código.")

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trilha de Competências")

# Inicialização do estado da sessão
if 'trilha' not in st.session_state:
    st.session_state.trilha = {}

COMPETENCIAS = {
    "competencia1": "Domínio da Norma Culta",
    "competencia2": "Compreensão do Tema",
    "competencia3": "Seleção e Organização das Informações",
    "competencia4": "Conhecimento dos Mecanismos Linguísticos",
    "competencia5": "Proposta de Intervenção"
}

def apresentar_competencia(competencia: str):
    """Apresenta a competência selecionada."""
    st.subheader(f"Competência: {COMPETENCIAS[competencia]}")
    st.write(f"Nesta etapa, analisaremos a {COMPETENCIAS[competencia]}.")

def identificar_agrupamento_erros(competencia: str, erros_extraidos: List[str]):
    """Identifica e agrupa os erros extraídos pela análise."""
    st.subheader("Identificação e Agrupamento de Erros")
    if erros_extraidos:
        st.write("### Erros Detectados:")
        for erro in erros_extraidos:
            st.markdown(f"- {erro}")
    else:
        st.write("Nenhum erro detectado.")

def teoria_exercicios_personalizados(erros_detectados: List[str]):
    """Apresenta teoria e exercícios personalizados para os erros detectados."""
    st.subheader("Teoria e Exercícios Personalizados")
    for erro in erros_detectados:
        st.write(f"**Erro:** {erro}")
        st.write(f"Teoria sobre {erro}:")
        st.write("Aqui você encontrará explicações detalhadas e exemplos sobre como corrigir esse tipo de erro.")
        st.write(f"Exercício: Corrija a frase abaixo que contém um erro de {erro}.")

def finalizar_competencia(competencia: str, progresso: Dict[str, any]):
    """Finaliza a análise de uma competência e salva o progresso."""
    st.session_state.trilha[competencia] = progresso
    st.success(f"A competência {COMPETENCIAS[competencia]} foi concluída com sucesso!")

def processar_redacao(competencia: str, texto_redacao: str) -> List[str]:
    """Chama a função de análise correspondente para processar a redação."""
    try:
        from analysis_function import processar_redacao_completa
        resultados = processar_redacao_completa(texto_redacao, competencia)
        return resultados.get('erros', [])
    except ImportError as e:
        st.error("Erro ao importar a função de análise. Verifique se o arquivo 'analysis_function.py' está correto.")
        logger.error(f"Erro ao importar função de análise: {e}")
        return []
    except Exception as e:
        st.error("Erro ao processar a redação. Tente novamente.")
        logger.error(f"Erro ao processar redação: {e}")
        return []

def trilha_de_competencias():
    """Interface principal para trilha de competências."""
    st.title("Trilha de Competências ENEM")

    texto_redacao = st.text_area("Digite a redação para análise:", height=300)

    if not texto_redacao.strip():
        st.warning("Por favor, insira o texto da redação para começar.")
        return

    competencia_selecionada = st.selectbox("Escolha a competência para análise:", options=list(COMPETENCIAS.keys()), format_func=lambda x: COMPETENCIAS[x])

    if st.button("Iniciar Análise"):
        apresentar_competencia(competencia_selecionada)
        erros_detectados = processar_redacao(competencia_selecionada, texto_redacao)
        identificar_agrupamento_erros(competencia_selecionada, erros_detectados)
        teoria_exercicios_personalizados(erros_detectados)
        progresso = {"erros": erros_detectados, "texto": texto_redacao}
        finalizar_competencia(competencia_selecionada, progresso)

if __name__ == "__main__":
    trilha_de_competencias()

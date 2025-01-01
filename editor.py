import os
import streamlit as st
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
from anthropic import Anthropic
from elevenlabs import set_api_key, generate

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
except Exception as e:
    logger.error(f"Erro na inicialização do cliente Anthropic: {e}")
    st.error("Erro ao inicializar conexões. Por favor, tente novamente mais tarde.")

# Configuração da ElevenLabs
try:
    set_api_key(st.secrets["elevenlabs"]["api_key"])
except Exception as e:
    logger.error(f"Erro ao inicializar ElevenLabs: {e}")
    st.error("Erro ao configurar a API ElevenLabs.")

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

def processar_redacao_com_ia(texto: str, tema: str) -> Dict[str, Any]:
    """Processa a redação usando a API da Anthropic."""
    prompt = f"""
    Tema: {tema}
    Redação:
    {texto}
    
    Analise a redação acima de acordo com as competências do ENEM:
    - Competência 1: Domínio da Norma Culta
    - Competência 2: Compreensão do Tema
    - Competência 3: Seleção e Organização das Informações
    - Competência 4: Conhecimento dos Mecanismos Linguísticos
    - Competência 5: Proposta de Intervenção

    Para cada competência, forneça:
    1. Uma nota de 0 a 200.
    2. Justificativa da nota.
    3. Trechos com erros específicos (se houver).
    """
    try:
        response = anthropic_client.messages.create(
            model="claude-3",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response["completion"])
    except Exception as e:
        logger.error(f"Erro ao processar redação com IA: {e}")
        return {}

def pagina_envio_redacao():
    """Página principal de envio de redação"""
    st.title("Sistema de Análise de Redação ENEM")

    tema_redacao = st.text_input("Tema da redação:")
    texto_redacao = st.text_area("Digite sua redação aqui:", height=400)

    if st.button("Analisar Redação"):
        if tema_redacao and texto_redacao:
            with st.spinner("Analisando redação..."):
                resultados = processar_redacao_com_ia(texto_redacao, tema_redacao)
                if resultados:
                    st.session_state.resultados = resultados
                    st.session_state.tema_redacao = tema_redacao
                    st.session_state.texto_redacao = texto_redacao
                    st.session_state.page = 'resultado'
                    st.experimental_rerun()
                else:
                    st.error("Erro ao processar a redação.")
        else:
            st.warning("Por favor, insira o tema e o texto da redação.")

def pagina_resultado_analise():
    """Página de exibição dos resultados da análise"""
    st.title("Resultado da Análise")

    if 'resultados' not in st.session_state:
        st.warning("Nenhuma análise disponível. Por favor, envie uma redação.")
        return

    resultados = st.session_state.resultados
    tema_redacao = st.session_state.tema_redacao

    st.subheader(f"Tema: {tema_redacao}")

    for comp, details in resultados.items():
        st.markdown(f"### {COMPETENCIES.get(comp, 'Competência desconhecida')}")
        st.write(f"**Nota:** {details['nota']}/200")
        st.write(f"**Justificativa:** {details['justificativa']}")
        if details.get("erros"):
            st.markdown("#### Erros Identificados:")
            for erro in details["erros"]:
                st.write(f"- {erro}")

def main():
    """Função principal que controla o fluxo da aplicação"""
    if st.session_state.page == 'envio':
        pagina_envio_redacao()
    elif st.session_state.page == 'resultado':
        pagina_resultado_analise()

if __name__ == "__main__":
    main()

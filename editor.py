import streamlit as st
from openai import OpenAI
import sys
from typing import Dict, Optional, List

# Configuração e verificação da API key
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if not client.api_key:
        st.error("Chave da API não encontrada!")
        sys.exit(1)
except Exception as e:
    st.error(f"Erro ao configurar a chave API: {e}")
    sys.exit(1)

class RedacaoAssistant:
    def __init__(self):
        self.system_prompt = """Você é um assistente especializado e amigável de redação do ENEM.
        Seu papel é conduzir uma conversa interativa com o estudante, ajudando-o a desenvolver sua redação passo a passo.
        Seja específico nas orientações e mantenha um tom encorajador."""

    async def chat_with_user(self, prompt: str, history: Optional[List[Dict]] = None) -> Dict:
        if history is None:
            history = []
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            return {"status": "success", "response": response.choices[0].message.content}
        except Exception as e:
            return {"status": "error", "message": str(e)}
st.set_page_config(
    page_title="Assistente de Redação ENEM",
    page_icon="📝",
    layout="wide"
)

# Inicialização do estado
if 'stage' not in st.session_state:
    st.session_state.stage = 'inicio'
if 'assistant' not in st.session_state:
    st.session_state.assistant = RedacaoAssistant()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'redacao' not in st.session_state:
    st.session_state.redacao = {
        'tema': '',
        'introducao': '',
        'desenvolvimento1': '',
        'desenvolvimento2': '',
        'conclusao': ''
    }

# Interface Principal
st.title("📝 Assistente Interativo de Redação ENEM")

# Sidebar com progresso
st.sidebar.title("Progresso da Redação")
progress_items = {
    'inicio': 'Planejamento Inicial',
    'introducao': 'Introdução',
    'desenvolvimento1': '1º Parágrafo',
    'desenvolvimento2': '2º Parágrafo',
    'conclusao': 'Conclusão',
    'revisao': 'Revisão Final'
}

for key, value in progress_items.items():
    if st.session_state.stage == key:
        st.sidebar.markdown(f"**→ {value}**")
    elif list(progress_items.keys()).index(key) < list(progress_items.keys()).index(st.session_state.stage):
        st.sidebar.markdown(f"✅ {value}")
    else:
        st.sidebar.markdown(f"◽ {value}")

# Função para chat
async def get_assistant_response(prompt):
    response = await st.session_state.assistant.chat_with_user(
        prompt,
        st.session_state.chat_history
    )
    if response["status"] == "success":
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response["response"]})
        return response["response"]
    return "Desculpe, houve um erro. Tente novamente."

# Área principal - Diferentes estágios da redação
if st.session_state.stage == 'inicio':
    st.markdown("### 🎯 Vamos começar sua redação!")
    
    if not st.session_state.redacao['tema']:
        tema_input = st.text_input("Sobre qual tema você quer escrever?")
        if tema_input:
            st.session_state.redacao['tema'] = tema_input
            response = asyncio.run(get_assistant_response(
                f"O tema da redação é: {tema_input}. Me ajude a planejar essa redação, sugerindo possíveis argumentos e repertório sociocultural relevante."
            ))
            st.markdown(response)
    
    if st.session_state.redacao['tema']:
        st.markdown(f"**Tema escolhido:** {st.session_state.redacao['tema']}")
        
        user_input = st.text_input("Pode me fazer perguntas sobre o tema ou pedir sugestões")
        if user_input:
            response = asyncio.run(get_assistant_response(user_input))
            st.markdown(response)
        
        if st.button("Começar a escrever a introdução"):
            st.session_state.stage = 'introducao'
            st.rerun()

elif st.session_state.stage == 'introducao':
    st.markdown("### 📝 Introdução")
    st.markdown("Escreva um parágrafo introdutório que contextualize o tema e apresente sua tese.")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        introducao = st.text_area("Seu parágrafo introdutório:", 
                                value=st.session_state.redacao['introducao'],
                                height=200)
        if introducao != st.session_state.redacao['introducao']:
            st.session_state.redacao['introducao'] = introducao
            if introducao:
                response = asyncio.run(get_assistant_response(
                    f"Analise este parágrafo introdutório: {introducao}"
                ))
                st.session_state.last_feedback = response
    
    with col2:
        if 'last_feedback' in st.session_state:
            st.markdown("### Feedback")
            st.markdown(st.session_state.last_feedback)
    
    if st.button("Avançar para o desenvolvimento"):
        st.session_state.stage = 'desenvolvimento1'
        st.rerun()

elif st.session_state.stage == 'desenvolvimento1':
    st.markdown("### 📝 Primeiro Parágrafo de Desenvolvimento")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        desenvolvimento1 = st.text_area("Desenvolva seu primeiro argumento:", 
                                      value=st.session_state.redacao['desenvolvimento1'],
                                      height=200)
        if desenvolvimento1 != st.session_state.redacao['desenvolvimento1']:
            st.session_state.redacao['desenvolvimento1'] = desenvolvimento1
            if desenvolvimento1:
                response = asyncio.run(get_assistant_response(
                    f"Analise este parágrafo de desenvolvimento: {desenvolvimento1}"
                ))
                st.session_state.last_feedback = response
    
    with col2:
        if 'last_feedback' in st.session_state:
            st.markdown("### Feedback")
            st.markdown(st.session_state.last_feedback)
    
    if st.button("Avançar para o segundo parágrafo"):
        st.session_state.stage = 'desenvolvimento2'
        st.rerun()

elif st.session_state.stage == 'desenvolvimento2':
    st.markdown("### 📝 Segundo Parágrafo de Desenvolvimento")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        desenvolvimento2 = st.text_area("Desenvolva seu segundo argumento:", 
                                      value=st.session_state.redacao['desenvolvimento2'],
                                      height=200)
        if desenvolvimento2 != st.session_state.redacao['desenvolvimento2']:
            st.session_state.redacao['desenvolvimento2'] = desenvolvimento2
            if desenvolvimento2:
                response = asyncio.run(get_assistant_response(
                    f"Analise este segundo parágrafo: {desenvolvimento2}"
                ))
                st.session_state.last_feedback = response
    
    with col2:
        if 'last_feedback' in st.session_state:
            st.markdown("### Feedback")
            st.markdown(st.session_state.last_feedback)
    
    if st.button("Avançar para a conclusão"):
        st.session_state.stage = 'conclusao'
        st.rerun()

elif st.session_state.stage == 'conclusao':
    st.markdown("### 📝 Conclusão")
    st.markdown("Apresente sua proposta de intervenção.")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        conclusao = st.text_area("Escreva sua conclusão:", 
                               value=st.session_state.redacao['conclusao'],
                               height=200)
        if conclusao != st.session_state.redacao['conclusao']:
            st.session_state.redacao['conclusao'] = conclusao
            if conclusao:
                response = asyncio.run(get_assistant_response(
                    f"Analise esta conclusão: {conclusao}"
                ))
                st.session_state.last_feedback = response
    
    with col2:
        if 'last_feedback' in st.session_state:
            st.markdown("### Feedback")
            st.markdown(st.session_state.last_feedback)
    
    if st.button("Fazer revisão final"):
        st.session_state.stage = 'revisao'
        st.rerun()

elif st.session_state.stage == 'revisao':
    st.markdown("### 📋 Revisão Final")
    
    texto_completo = f"""
    {st.session_state.redacao['introducao']}

    {st.session_state.redacao['desenvolvimento1']}

    {st.session_state.redacao['desenvolvimento2']}

    {st.session_state.redacao['conclusao']}
    """
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("### Sua Redação Completa")
        st.text_area("Texto final:", value=texto_completo, height=400)
    
    with col2:
        if st.button("Analisar redação completa"):
            response = asyncio.run(get_assistant_response(
                f"Faça uma análise completa desta redação, avaliando todas as competências do ENEM: {texto_completo}"
            ))
            st.markdown("### Análise Final")
            st.markdown(response)

# Botão para recomeçar (sempre visível)
if st.sidebar.button("Recomeçar Redação"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Estatísticas na sidebar
if st.session_state.redacao['introducao']:
    total_words = len(' '.join([
        st.session_state.redacao['introducao'],
        st.session_state.redacao['desenvolvimento1'],
        st.session_state.redacao['desenvolvimento2'],
        st.session_state.redacao['conclusao']
    ]).split())
    
    st.sidebar.markdown("### Estatísticas")
    st.sidebar.metric("Total de palavras", total_words)

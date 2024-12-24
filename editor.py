import streamlit as st
import openai
import asyncio
from typing import Dict

# Configuração do OpenAI usando o secret do Streamlit
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Configuração da página
st.set_page_config(
    page_title="Assistente de Redação ENEM",
    page_icon="📝",
    layout="wide"
)

class RedacaoAnalyzer:
    def __init__(self):
        self.system_prompt = """Você é um assistente especializado em redação do ENEM.
        Analise o texto considerando as 5 competências:
        
        1. Domínio da norma culta
        2. Compreensão do tema e estrutura dissertativa
        3. Argumentação
        4. Coesão textual
        5. Proposta de intervenção
        
        Para cada parágrafo, forneça:
        - Pontos positivos
        - Sugestões de melhoria
        - Dicas específicas para aprimoramento
        
        Use uma linguagem amigável e construtiva."""

    async def analyze_text(self, text: str) -> Dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Analise o seguinte texto para a redação do ENEM: {text}"}
        ]

        try:
            response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            return {"status": "success", "feedback": response.choices[0].message.content}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_competency_scores(self, text: str) -> Dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Avalie as 5 competências do ENEM para o seguinte texto, 
            dando uma nota de 0 a 200 para cada e uma breve justificativa: {text}"""}
        ]

        try:
            response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            return {"status": "success", "scores": response.choices[0].message.content}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Inicialização do estado da sessão
if 'redacao_text' not in st.session_state:
    st.session_state.redacao_text = ""
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = RedacaoAnalyzer()
if 'tema' not in st.session_state:
    st.session_state.tema = ""

# Função para atualizar o feedback
async def update_feedback():
    if st.session_state.redacao_text:
        feedback = await st.session_state.analyzer.analyze_text(st.session_state.redacao_text)
        if feedback["status"] == "success":
            st.session_state.feedback = feedback["feedback"]
        else:
            st.session_state.feedback = "Erro na análise. Tente novamente."

# Interface principal
st.title("📝 Assistente de Redação ENEM")

# Área do tema
st.session_state.tema = st.text_input("Digite o tema da redação:", st.session_state.tema)

# Layout em duas colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sua Redação")
    redacao_text = st.text_area(
        "Digite sua redação aqui",
        height=400,
        key="redacao_input",
        on_change=lambda: asyncio.run(update_feedback())
    )

with col2:
    st.subheader("Feedback do Assistente")
    if st.session_state.feedback:
        st.markdown(st.session_state.feedback)

    if st.button("Analisar Competências"):
        with st.spinner("Analisando competências..."):
            scores = asyncio.run(
                st.session_state.analyzer.get_competency_scores(st.session_state.redacao_text)
            )
            if scores["status"] == "success":
                st.markdown("### Notas por Competência")
                st.markdown(scores["scores"])

# Área de estatísticas
st.sidebar.title("Estatísticas")
if st.session_state.redacao_text:
    word_count = len(st.session_state.redacao_text.split())
    char_count = len(st.session_state.redacao_text)
    paragraph_count = len(st.session_state.redacao_text.split('\n\n')) + 1
    
    st.sidebar.metric("Palavras", word_count)
    st.sidebar.metric("Caracteres", char_count)
    st.sidebar.metric("Parágrafos", paragraph_count)
    
    # Avaliação rápida do tamanho
    if word_count < 250:
        st.sidebar.warning("⚠️ Texto muito curto. Procure desenvolver mais.")
    elif word_count > 350:
        st.sidebar.warning("⚠️ Texto muito longo. Considere sintetizar.")
    else:
        st.sidebar.success("✅ Tamanho adequado!")

# Dicas e recomendações
st.sidebar.markdown("""
### Dicas para Nota 1000
1. **Introdução**:
   - Contextualize o tema
   - Apresente sua tese
   
2. **Desenvolvimento**:
   - Use repertório sociocultural
   - Conecte os parágrafos
   
3. **Conclusão**:
   - Proposta de intervenção completa
   - Retome aspectos principais
""")

# Área de ajuda
with st.expander("Como usar o assistente"):
    st.markdown("""
    ### Instruções de Uso
    
    1. **Digite o tema** no campo superior
    2. **Escreva sua redação** no editor principal
    3. Receba **feedback em tempo real**
    4. Use o botão **Analisar Competências** para avaliação detalhada
    5. Acompanhe as **estatísticas** na barra lateral
    
    ### Estrutura Recomendada
    
    - **Introdução**: 1 parágrafo
    - **Desenvolvimento**: 2-3 parágrafos
    - **Conclusão**: 1 parágrafo
    
    ### Critérios de Avaliação
    
    1. Domínio da norma culta (200 pontos)
    2. Compreensão do tema (200 pontos)
    3. Argumentação (200 pontos)
    4. Coesão textual (200 pontos)
    5. Proposta de intervenção (200 pontos)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desenvolvido para auxiliar estudantes na preparação para o ENEM</p>
</div>
""", unsafe_allow_html=True)

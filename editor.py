import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

class ConteudoLinguagens:
    def __init__(self):
        self.temas = {
            "Gêneros Textuais": {"frequencia": 41, "facil": 21, "medio": 16, "dificil": 4},
            "Textos Não Literários": {"frequencia": 34, "facil": 18, "medio": 14, "dificil": 2},
            "Noções Básicas de Compreensão de Texto": {"frequencia": 32, "facil": 17, "medio": 12, "dificil": 3},
            "Textos Literários": {"frequencia": 32, "facil": 6, "medio": 8, "dificil": 18},
            "Variações Linguísticas": {"frequencia": 25, "facil": 12, "medio": 9, "dificil": 4},
            "Arte e Literatura": {"frequencia": 22, "facil": 2, "medio": 8, "dificil": 12},
            "Tecnologias da Comunicação": {"frequencia": 18, "facil": 9, "medio": 7, "dificil": 2}
        }

def main():
    st.set_page_config(page_title="ENEM Linguagens - Plano de Estudos", layout="wide")
    st.title("📚 Plano de Estudos ENEM - Linguagens")
    
    conteudo = ConteudoLinguagens()
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.subheader("Distribuição de Temas por Nível")
        df = pd.DataFrame(conteudo.temas).T
        fig = px.bar(df, 
                    x=df.index, 
                    y=['facil', 'medio', 'dificil'],
                    title="Distribuição de Questões por Nível de Dificuldade",
                    labels={'value': 'Quantidade', 'variable': 'Nível'},
                    height=500)
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Prioridades de Estudo")
        prioridades = pd.DataFrame(conteudo.temas).T.sort_values('frequencia', ascending=False)
        st.write("Baseado na frequência de aparição:")
        for tema, row in prioridades.iterrows():
            st.write(f"• {tema}: {int(row['frequencia'])} questões")
    
    st.subheader("Cronograma Sugerido")
    dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    selected_day = st.selectbox("Selecione o dia", dias_semana)
    
    if selected_day:
        st.write(f"### Plano de Estudos para {selected_day}")
        st.write("Tempo total disponível: 1h30")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.info("30min - Conteúdo Novo")
        with col4:
            st.warning("30min - Exercícios")
        with col5:
            st.success("30min - Revisão")

if __name__ == "__main__":
    main()

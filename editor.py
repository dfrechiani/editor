import streamlit as st
import pandas as pd
import altair as alt
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

def criar_cronograma_semanal():
    cronograma = {
        "Segunda": {
            "Tema Principal": "Gêneros Textuais",
            "Exercícios": "Questões de nível fácil",
            "Revisão": "Noções Básicas de Compreensão"
        },
        "Terça": {
            "Tema Principal": "Textos Não Literários",
            "Exercícios": "Questões de nível médio",
            "Revisão": "Gêneros Textuais"
        },
        "Quarta": {
            "Tema Principal": "Textos Literários",
            "Exercícios": "Questões de nível difícil",
            "Revisão": "Textos Não Literários"
        },
        "Quinta": {
            "Tema Principal": "Variações Linguísticas",
            "Exercícios": "Questões mistas",
            "Revisão": "Textos Literários"
        },
        "Sexta": {
            "Tema Principal": "Arte e Literatura",
            "Exercícios": "Questões de nível médio/difícil",
            "Revisão": "Variações Linguísticas"
        },
        "Sábado": {
            "Tema Principal": "Tecnologias da Comunicação",
            "Exercícios": "Simulado",
            "Revisão": "Arte e Literatura"
        },
        "Domingo": {
            "Tema Principal": "Revisão Geral",
            "Exercícios": "Redação",
            "Revisão": "Temas da semana"
        }
    }
    return cronograma

def main():
    st.set_page_config(page_title="ENEM Linguagens - Plano de Estudos", layout="wide")
    st.title("📚 Plano de Estudos ENEM - Linguagens")
    
    conteudo = ConteudoLinguagens()
    
    # Criando DataFrame para visualização
    df = pd.DataFrame(conteudo.temas).T
    df = df.sort_values('frequencia', ascending=False)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.subheader("Análise dos Temas")
        chart_data = df.reset_index()
        chart_data = chart_data.rename(columns={'index': 'Tema'})
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Tema',
            y='frequencia',
            color=alt.value('#1f77b4')
        ).properties(
            height=400
        )
        
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.subheader("Ranking de Prioridades")
        for tema, row in df.iterrows():
            st.write(f"• {tema}: {int(row['frequencia'])} questões")
    
    st.subheader("Cronograma Semanal")
    cronograma = criar_cronograma_semanal()
    dias_semana = list(cronograma.keys())
    selected_day = st.selectbox("Selecione o dia", dias_semana)
    
    if selected_day:
        st.write(f"### Plano de Estudos para {selected_day}")
        st.write("Tempo total disponível: 1h30")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.info(f"30min - Conteúdo Novo\n{cronograma[selected_day]['Tema Principal']}")
        with col4:
            st.warning(f"30min - Exercícios\n{cronograma[selected_day]['Exercícios']}")
        with col5:
            st.success(f"30min - Revisão\n{cronograma[selected_day]['Revisão']}")
            
        st.write("---")
        st.write("#### Dicas para o dia:")
        if selected_day == "Domingo":
            st.write("• Faça um resumo do que aprendeu durante a semana")
            st.write("• Dedique tempo extra para redação")
            st.write("• Revise os pontos que teve mais dificuldade")
        else:
            st.write("• Faça anotações durante o estudo")
            st.write("• Resolva pelo menos 5 questões do tema do dia")
            st.write("• Use técnicas de revisão ativa (explicar o conteúdo, fazer mapas mentais)")

if __name__ == "__main__":
    main()

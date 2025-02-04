import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime, timedelta

st.set_page_config(page_title="ENEM Linguagens - Plano de Estudos", layout="wide")  # Deve ser a primeira linha!

# 🔍 Carregar a chave corretamente
openai_api_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ A chave da API OpenAI não foi encontrada. Verifique `Manage app > Secrets` no Streamlit Cloud.")
    st.stop()  # ⛔ Para a execução do script se a chave não estiver definida
else:
    st.success("✅ Chave da API carregada com sucesso!")

# ✅ Definir a chave diretamente na configuração do OpenAI
openai.api_key = openai_api_key  # O OpenAI agora lê a chave diretamente assim





class BancoQuestoesEnem:
   def __init__(self):
       self.generos_textuais = {
           "faceis": [
               {
                   "id": "GT001", 
                   "ano": 2019,
                   "texto": "Na sociologia e na literatura, o brasileiro foi por vezes tratado como cordial...",
                   "alternativas": [
                       "minimiza o alcance da comunicação digital",
                       "refuta ideias preconcebidas sobre o brasileiro", 
                       "relativiza responsabilidades sobre a noção de respeito",
                       "exemplifica conceitos contidos na literatura e na sociologia",
                       "expõe a ineficácia dos estudos para alterar tal comportamento"
                   ],
                   "gabarito": "B",
                   "dificuldade": "Fácil",
                   "habilidades": ["Interpretação", "Análise crítica", "Compreensão textual"],
                   "explicacao": "O texto contrapõe a visão tradicional do brasileiro cordial...",
                   "tema_especifico": "Texto jornalístico/reportagem"
               },
               {
                   "id": "GT002",
                   "ano": 2019,
                   "texto": "O rio que fazia uma volta atrás de nossa casa...",
                   "alternativas": [
                       "a terminologia mencionada é incorreta",
                       "a nomeação minimiza a percepção subjetiva",
                       "a palavra é aplicada a outro espaço geográfico",
                       "a designação atribuída ao termo é desconhecida",
                       "a definição modifica o significado do termo no dicionário"
                   ],
                   "gabarito": "B",
                   "dificuldade": "Fácil", 
                   "habilidades": ["Compreensão literária", "Análise semântica", "Interpretação poética"],
                   "explicacao": "O texto de Manoel de Barros trabalha com a oposição...",
                   "tema_especifico": "Poesia moderna"
               }
           ],
           "medias": [
               {
                   "id": "GT010",
                   "ano": 2020,
                   "texto": "Mulher tem coração clinicamente partido após morte de cachorro...",
                   "alternativas": [
                       "conto, pois exibe a história de vida de Joanie Simpson",
                       "depoimento, pois expõe o sofrimento da dona do animal",
                       "reportagem, pois discute cientificamente a cardiomiopatia",
                       "relato, pois narra um fato estressante vivido pela paciente",
                       "notícia, pois divulga fatos sobre a síndrome do coração partido"
                   ],
                   "gabarito": "E",
                   "dificuldade": "Média",
                   "habilidades": ["Identificação de gêneros", "Análise textual"],
                   "explicacao": "A questão avalia a capacidade de identificar características do gênero notícia...",
                   "tema_especifico": "Gêneros jornalísticos"
               }
           ],
           "dificeis": [
               {
                   "id": "GT020",
                   "ano": 2021,
                   "texto": "Ed Mort só vai... Mort. Ed Mort. Detetive particular...", 
                   "alternativas": [
                       "segmentação de enunciados baseada na descrição dos hábitos do personagem",
                       "ordenação dos constituintes oracionais na qual se destaca o núcleo verbal",
                       "estrutura composicional caracterizada pelo arranjo singular dos períodos",
                       "sequenciação narrativa na qual se articulam eventos absurdos",
                       "seleção lexical na qual predominam informações redundantes"
                   ],
                   "gabarito": "D",
                   "dificuldade": "Difícil",
                   "habilidades": ["Análise estilística", "Compreensão narrativa"],
                   "explicacao": "O texto utiliza uma estrutura narrativa que articula eventos...",
                   "tema_especifico": "Narrativa contemporânea"
               }
           ]
       }
       
       self.textos_nao_literarios = {
           "faceis": [
               {
                   "id": "TNL001",
                   "ano": 2019,
                   "texto": "Em 2000 tivemos a primeira experiência do futebol feminino em um jogo de videogame...",
                   "alternativas": [
                       "disseminarem uma modalidade, promovendo a igualdade de gênero",
                       "superarem jogos malsucedidos no mercado, lançados anteriormente", 
                       "inovarem a modalidade com novas ofertas de jogos ao mercado",
                       "explorarem nichos de mercado antes ignorados, produzindo mais lucro",
                       "reforçarem estereótipos de gênero masculino ou feminino nos esportes"
                   ],
                   "gabarito": "A",
                   "dificuldade": "Fácil",
                   "habilidades": ["Interpretação textual", "Análise crítica", "Compreensão de argumentação"],
                   "explicacao": "O texto mostra como a inclusão do futebol feminino nos jogos eletrônicos contribui para disseminar a modalidade...",
                   "tema_especifico": "Texto informativo/argumentativo"
               },
               {
                   "id": "TNL002", 
                   "ano": 2020,
                   "texto": "Reaprender a ler notícias. Não dá mais para ler um jornal...",
                   "alternativas": [
                       "buscarem fontes de informação comprometidas com a verdade",
                       "privilegiarem notícias veiculadas em jornais de grande circulação",
                       "adotarem uma postura crítica em relação às informações recebidas",
                       "questionarem a prática jornalística anterior ao surgimento da internet",
                       "valorizarem reportagens redigidas com imparcialidade diante dos fatos"
                   ],
                   "gabarito": "C",
                   "dificuldade": "Fácil",
                   "habilidades": ["Leitura crítica", "Análise de mídia", "Compreensão textual"],
                   "explicacao": "O texto argumenta sobre a necessidade de uma leitura mais crítica e analítica das notícias...",
                   "tema_especifico": "Texto jornalístico/mídia"
               }
           ],
           "medias": [
               {
                   "id": "TNL010",
                   "ano": 2021,
                   "texto": "Uma das mais contundentes críticas ao discurso da aptidão física...",
                   "alternativas": [
                       "constroi a ideia de que a mudança individual de hábitos promove a saúde",
                       "considera a homogeneidade da escolha de hábitos saudáveis pelos indivíduos",
                       "reforça a necessidade de solucionar os problemas de saúde da sociedade com a prática de exercícios",
                       "problematiza a organização social e seu impacto na mudança de hábitos dos indivíduos",
                       "reproduz a noção de que a melhoria da aptidão física pela prática de exercícios promove a saúde"
                   ],
                   "gabarito": "D",
                   "dificuldade": "Média",
                   "habilidades": ["Análise argumentativa", "Compreensão crítica", "Interpretação textual"],
                   "explicacao": "O texto critica a visão individualista da saúde e aptidão física...",
                   "tema_especifico": "Texto argumentativo/científico"
               }
           ],
           "dificeis": [
               {
                   "id": "TNL020",
                   "ano": 2022,
                   "texto": "O complexo de falar difícil...", 
                   "alternativas": [
                       "se ter um notável saber jurídico",
                       "valorização da inteligência do falante",
                       "falar difícil para demonstrar inteligência",
                       "coesão e da coerência em documentos jurídicos",
                       "adequação da linguagem à situação de comunicação"
                   ],
                   "gabarito": "E",
                   "dificuldade": "Difícil",
                   "habilidades": ["Análise linguística", "Compreensão sociolinguística", "Reflexão crítica"],
                   "explicacao": "O texto discute a importância da adequação linguística ao contexto comunicativo...",
                   "tema_especifico": "Texto crítico/reflexivo"
               }
           ]
       }
       
       self.compreensao_textual = {
           "faceis": [
               {
                   "id": "CT001",
                   "ano": 2020,
                   "texto": "Que coisas devo levar / nesta viagem em que partes?...",
                   "alternativas": [
                       "saudade como experiência de apatia",
                       "presença da fragmentação da identidade",
                       "negação do desejo como expressão de culpa",
                       "persistência da memória na valorização do passado",
                       "revelação de rumos projetada pela vivência da solidão"
                   ],
                   "gabarito": "E",
                   "dificuldade": "Fácil",
                   "habilidades": ["Interpretação poética", "Análise de sentimentos", "Compreensão metafórica"],
                   "explicacao": "O poema trabalha com a metáfora da viagem para expressar a solidão...",
                   "tema_especifico": "Interpretação poética"
               },
               {
                   "id": "CT002",
                   "ano": 2019,
                   "texto": "Entre a toalha branca e um bule de café...",
                   "alternativas": [
                       "invocar o interlocutor para uma tomada de posição",
                       "questionar a validade do envolvimento romântico",
                       "diluir em banalidade a comoção de um amor frustrado",
                       "transformar em paz as emoções conflituosas do casal",
                       "condicionar a existência da paixão a espaços idealizados"
                   ],
                   "gabarito": "C",
                   "dificuldade": "Fácil",
                   "habilidades": ["Interpretação literária", "Análise de sentimentos", "Compreensão contextual"],
                   "explicacao": "O poema contrasta a grandiosidade do momento de término...",
                   "tema_especifico": "Interpretação poética" 
               }
           ],
           "medias": [
               {
                   "id": "CT010",
                   "ano": 2021,
                   "texto": "Os velhos papéis, quando não são consumidos pelo fogo...",
                   "alternativas": [
                       "A vida às vezes é como um jogo brincado na rua",
                       "Há ocorrências bem singulares. Está vendo aquela dama?",
                       "Aquelas mulheres sentadas na varanda das casas",
                       "O tempo corria depressa quando a gente era criança",
                       "Os dias mágicos passam depressa deixando marcas fundas"
                   ],
                   "gabarito": "A",
                   "dificuldade": "Média",
                   "habilidades": ["Interpretação textual", "Análise comparativa", "Compreensão de analogias"],
                   "explicacao": "A questão avalia a capacidade de identificar a analogia entre a descoberta...",
                   "tema_especifico": "Compreensão de analogias"
               }
           ],
           "dificeis": [
               {
                   "id": "CT020",
                   "ano": 2022,
                   "texto": "Seus primeiros anos de detento foram difíceis...",
                   "alternativas": [
                       "buscam perpetuar visões do senso comum",
                       "trazem à tona atitudes de um estado de exceção",
                       "promovem a interlocução com grupos silenciados",
                       "inspiram o sentimento de justiça por meio da empatia",
                       "recorrem ao absurdo como forma de traduzir a realidade"
                   ],
                   "gabarito": "B",
                   "dificuldade": "Difícil", 
                   "habilidades": ["Análise crítica", "Compreensão contextual", "Interpretação social"],
                   "explicacao": "O texto utiliza recursos narrativos para evidenciar a violência institucionalizada...",
                   "tema_especifico": "Análise social crítica"
               }
           ]
       }

       self.textos_literarios = {
           "faceis": [
               {
                   "id": "TL001",
                   "ano": 2021,
                   "texto": "Sinhá. Se a dona se banhou...",
                   "alternativas": [
                       "remetem à violência física e simbólica contra os povos escravizados",
                       "valorizam as influências da cultura africana sobre a música nacional",
                       "relativizam o sincretismo constitutivo das práticas religiosas brasileiras",
                       "narram os infortúnios da relação amorosa entre membros de classes sociais diferentes",
                       "problematizam as diferentes visões de mundo na sociedade durante o período colonial"
                   ],
                   "gabarito": "A",
                   "dificuldade": "Fácil",
                   "habilidades": ["Interpretação literária", "Análise histórica", "Compreensão cultural"],
                   "explicacao": "O texto retrata a violência física e simbólica do período escravocrata...",
                   "tema_especifico": "Literatura e escravidão"
               }
           ],
           "medias": [
               {
                   "id": "TL010",
                   "ano": 2020,
                   "texto": "Garcia tinha-se chegado ao cadáver...",
                   "alternativas": [
                       "indignação face à suspeita do adultério da esposa",
                       "tristeza compartilhada pela perda da mulher amada",
                       "espanto diante da demonstração de afeto de Garcia",
                       "prazer da personagem em relação ao sofrimento alheio",
                       "superação do ciúme pela comoção decorrente da morte"
                   ],
                   "gabarito": "D",
                   "dificuldade": "Média",
                   "habilidades": ["Análise psicológica", "Compreensão narrativa", "Interpretação de personagens"],
                   "explicacao": "O texto de Machado de Assis explora o sadismo de Fortunato...",
                   "tema_especifico": "Literatura machadiana"
               }
           ],
           "dificeis": [
               {
                   "id": "TL020",
                   "ano": 2022,
                   "texto": "Romanos usavam redes sociais há dois mil anos, diz livro...",
                   "alternativas": [
                       "imediatismo das respostas",
                       "compartilhamento de informações",
                       "interferência direta de outros no texto original",
                       "recorrência de seu uso entre membros da elite",
                       "perfil social dos envolvidos na troca comunicativa"
                   ],
                   "gabarito": "B",
                   "dificuldade": "Difícil",
                   "habilidades": ["Análise comparativa", "Compreensão histórica", "Interpretação cultural"],
                   "explicacao": "O texto estabelece uma analogia entre as práticas comunicativas...",
                   "tema_especifico": "Literatura comparada/comunicação"
               }
           ]
       }

       self.variacoes_linguisticas = {
           "faceis": [
               {
                   "id": "VL001",
                   "ano": 2023,
                   "texto": "Mandioca, macaxeira, aipim e castelinha são nomes diferentes...",
                   "alternativas": [
                       "passa por fenômenos de variação linguística como qualquer outra língua",
                       "apresenta variações regionais, assumindo novo sentido para algumas palavras",
                       "sofre mudança estrutural motivada pelo uso de sinais diferentes para algumas palavras",
                       "diferencia-se em todo o Brasil, desenvolvendo cada região a sua própria língua de sinais",
                       "é ininteligível para parte dos usuários em razão das mudanças de sinais motivadas geograficamente"
                   ],
                   "gabarito": "A",
                   "dificuldade": "Fácil",
                   "habilidades": ["Identificação de variações", "Compreensão sociolinguística", "Análise comparativa"],
                   "explicacao": "O texto mostra que a Libras, assim como qualquer língua...",
                   "tema_especifico": "Variação linguística em Libras"
               }
           ],
           "medias": [
               {
                   "id": "VL010",
                   "ano": 2022,
                   "texto": "— Famigerado? [...] — Famigerado é 'inóxio'...",
                   "alternativas": [
                       "local de origem dos interlocutores",
                       "estado emocional dos interlocutores",
                       "grau de coloquialidade da comunicação",
                       "nível de intimidade entre os interlocutores",
                       "conhecimento compartilhado na comunicação"
                   ],
                   "gabarito": "C",
                   "dificuldade": "Média",
                   "habilidades": ["Análise da fala", "Compreensão sociolinguística", "Interpretação contextual"],
                   "explicacao": "O diálogo evidencia diferentes níveis de formalidade...",
                   "tema_especifico": "Variação diastrática"
               }
           ],
           "dificeis": [
               {
                   "id": "VL020",
                   "ano": 2021,
                   "texto": "De quem é esta língua? Uma pequena editora brasileira...",
                   "alternativas": [
                       "à dificuldade de consolidação da literatura brasileira em outros países",
                       "aos diferentes graus de instrução formal entre os falantes de língua portuguesa",
                       "à existência de uma língua ideal que alguns falantes lusitanos creem ser a falada em Portugal",
                       "ao intercâmbio cultural que ocorre entre os povos dos diferentes países de língua portuguesa",
                       "à distância territorial entre os falantes do português que vivem em Portugal e no Brasil"
                   ],
                   "gabarito": "C",
                   "dificuldade": "Difícil",
                   "habilidades": ["Análise sociolinguística", "Compreensão cultural", "Interpretação crítica"],
                   "explicacao": "O texto aborda o preconceito linguístico dos portugueses...",
                   "tema_especifico": "Preconceito linguístico"
               }
           ]
       }

       self.metadata = {
           "textos_literarios": {
               "niveis": {
                   "facil": 6,
                   "medio": 8,
                   "dificil": 18
               },
               "habilidades_avaliadas": [
                   "Interpretação literária",
                   "Análise estilística", 
                   "Compreensão narrativa",
                   "Análise psicológica",
                   "Interpretação cultural",
                   "Análise histórica"
               ]
           },
           "variacoes_linguisticas": {
               "niveis": {
                   "facil": 12,
                   "medio": 9,
                   "dificil": 4
               },
               "tipos_variacao": {
                   "diatopica": "Variação geográfica",
                   "diastratica": "Variação social",
                   "diafasica": "Variação situacional",
                   "diacronica": "Variação histórica"
               }
           }
       }

   def get_questoes_por_tema(self, tema, dificuldade=None, quantidade=None):
       categorias = {
           "Gêneros Textuais": self.generos_textuais,
           "Textos Não Literários": self.textos_nao_literarios,
           "Compreensão Textual": self.compreensao_textual,
           "Textos Literários": self.textos_literarios,
           "Variações Linguísticas": self.variacoes_linguisticas
       }
       
       if tema not in categorias:
           return []
           
       questoes = []
       categoria = categorias[tema]
       
       if dificuldade:
           dificuldade = dificuldade.lower()
           if dificuldade == "fácil":
               questoes.extend(categoria["faceis"])
           elif dificuldade == "média":
               questoes.extend(categoria["medias"])
           elif dificuldade == "difícil":
               questoes.extend(categoria["dificeis"])
       else:
           for nivel in ["faceis", "medias", "dificeis"]:
               questoes.extend(categoria[nivel])
               
       if quantidade:
           questoes = questoes[:quantidade]
           
       return questoes

   def get_questoes_por_habilidade(self, habilidade):
       todas_questoes = []
       categorias = [self.generos_textuais, self.textos_nao_literarios, 
                    self.compreensao_textual, self.textos_literarios, 
                    self.variacoes_linguisticas]
       
       for categoria in categorias:
           for nivel in ["faceis", "medias", "dificeis"]:
               for questao in categoria[nivel]:
                   if habilidade in questao["habilidades"]:
                       todas_questoes.append(questao)
       
       return todas_questoes

   def get_metadata(self, categoria):
       return self.metadata.get(categoria, {})

class GeradorConteudo:
   def __init__(self):
       self.model = "o3-mini"
       
   def gerar_material_estudo(self, tema, questoes, nivel_profundidade="alto"):
       prompt = self._criar_prompt_estudo(tema, questoes, nivel_profundidade)
       return self._fazer_requisicao(prompt)
       
   def gerar_dicas_resolucao(self, questao):
       prompt = self._criar_prompt_resolucao(questao)
       return self._fazer_requisicao(prompt)
       
   def _criar_prompt_estudo(self, tema, questoes, nivel_profundidade):
       exemplos_questoes = "\n".join([
           f"Questão {i+1}:\n{q['texto']}\n" +
           "Habilidades avaliadas: " + ", ".join(q['habilidades'])
           for i, q in enumerate(questoes)
       ])
       
       return f"""
       Crie um material de estudo aprofundado sobre {tema} para o ENEM, considerando as seguintes questões como referência:
       {exemplos_questoes}
       O material deve incluir:
       1. CONTEXTUALIZAÇÃO
       - Importância do tema no ENEM
       - Principais abordagens nas provas
       
       2. FUNDAMENTOS TEÓRICOS
       - Conceitos essenciais aprofundados
       - Conexões com outros temas relevantes
       
       3. ASPECTOS PRÁTICOS
       - Estratégias de identificação e análise
       - Armadilhas comuns nas questões
       
       4. EXEMPLOS CONTEXTUALIZADOS
       - Análise detalhada de casos
       - Explicação do raciocínio necessário
       
       5. EXERCÍCIOS GUIADOS
       - Resolução comentada passo a passo
       - Identificação das habilidades trabalhadas
       """
       
   def _criar_prompt_resolucao(self, questao):
       return f"""
       Analise a seguinte questão do ENEM:
       {questao['texto']}
       Forneça:
       1. Identificação da habilidade principal avaliada
       2. Conceitos-chave necessários
       3. Estratégia passo a passo de resolução
       4. Explicação da alternativa correta
       5. Por que as outras alternativas estão erradas
       6. Dicas para não cair em armadilhas similares
       """
       
   def _fazer_requisicao(self, prompt):
    try:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "Você é um professor especialista em preparação para o ENEM, "
                    "com vasta experiência em linguagens e suas tecnologias. "
                    "Forneça explicações profundas mas claras, usando exemplos práticos."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar conteúdo: {str(e)}"



def criar_estilo():
   return """
   <style>
       .card {
           border-radius: 10px;
           padding: 20px;
           margin: 10px 0;
           box-shadow: 0 4px 6px rgba(0,0,0,0.1);
           background: white;
       }
       .tag {
           display: inline-block;
           padding: 5px 10px;
           border-radius: 15px;
           font-size: 12px;
           font-weight: bold;
           margin-bottom: 10px;
       }
       .conteudo { background: #e3f2fd; color: #1565c0; }
       .exercicios { background: #f3e5f5; color: #7b1fa2; }
       .revisao { background: #e8f5e9; color: #2e7d32; }
       .tempo { 
           float: right;
           color: #757575;
           font-size: 14px;
       }
   </style>
   """

def criar_card_estudo(tag, titulo, descricao, tempo="30min"):
   tag_class = tag.lower()
   return f"""
   <div class="card">
       <span class="tag {tag_class}">{tag}</span>
       <span class="tempo">⏱️ {tempo}</span>
       <h3>{titulo}</h3>
       <p>{descricao}</p>
   </div>
   """

def main():   
   banco = BancoQuestoesEnem()
   gerador = GeradorConteudo()
   
   st.markdown(criar_estilo(), unsafe_allow_html=True)
   
   st.title("📚 Plano de Estudos ENEM - Linguagens")
   
   cronograma = {
       "Segunda": {
           "tema_principal": "Gêneros Textuais",
           "dificuldade_exercicios": "Fácil",
           "tema_revisao": "Noções Básicas de Compreensão"
       },
       "Terça": {
           "tema_principal": "Textos Não Literários",
           "dificuldade_exercicios": "Fácil",
           "tema_revisao": "Gêneros Textuais"
       },
       "Quarta": {
           "tema_principal": "Compreensão Textual",
           "dificuldade_exercicios": "Média",
           "tema_revisao": "Textos Não Literários"
       },
       "Quinta": {
           "tema_principal": "Textos Literários",
           "dificuldade_exercicios": "Média",
           "tema_revisao": "Compreensão Textual"
       },
       "Sexta": {
           "tema_principal": "Variações Linguísticas",
           "dificuldade_exercicios": "Difícil",
           "tema_revisao": "Textos Literários"
       }
   }
   
   dias = st.tabs(list(cronograma.keys()))
   
   for i, dia in enumerate(cronograma.keys()):
       with dias[i]:
           info_dia = cronograma[dia]
           
           questoes_dia = banco.get_questoes_por_tema(
               info_dia["tema_principal"], 
               info_dia["dificuldade_exercicios"]
           )
           
           st.markdown(criar_card_estudo(
               "CONTEÚDO",
               info_dia["tema_principal"],
               "Conceitos fundamentais e aplicações"
           ), unsafe_allow_html=True)
           
           st.markdown(criar_card_estudo(
               "EXERCÍCIOS",
               f"Questões de nível {info_dia['dificuldade_exercicios'].lower()}",
               f"Seleção de questões sobre {info_dia['tema_principal']}"
           ), unsafe_allow_html=True)
           
           st.markdown(criar_card_estudo(
               "REVISÃO",
               info_dia["tema_revisao"],
               "Revisão ativa e exercícios de fixação"
           ), unsafe_allow_html=True)
           
           with st.expander("📖 Material de Estudo"):
               if st.button(f"Gerar material sobre {info_dia['tema_principal']}", key=f"btn_{dia}"):
                   with st.spinner("Gerando material..."):
                       conteudo = gerador.gerar_material_estudo(
                           info_dia["tema_principal"],
                           questoes_dia[:3]
                       )
                       st.markdown(conteudo)
           
           with st.expander("📝 Questões do Dia"):
               for j, questao in enumerate(questoes_dia, 1):
                   st.subheader(f"Questão {j}")
                   st.write(questao["texto"])
                   for k, alt in enumerate(questao["alternativas"]):
                       st.write(f"{chr(65+k)}) {alt}")
                   
                   if st.button(f"Ver resposta {j}", key=f"resp_{dia}_{j}"):
                       st.success(f"Gabarito: {questao['gabarito']}")
                       st.info(questao["explicacao"])

if __name__ == "__main__":
   main()

from typing import Any, Dict, List
import logging
import streamlit as st


# Configuração básica do logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Agora você pode usar o logger no seu código
logger.debug("Mensagem de debug")
logger.info("Mensagem de informação")
logger.warning("Mensagem de aviso")


def processar_redacao_completa(redacao_texto: str, tema_redacao: Dict[str, Any]) -> Dict[str, Any]:
  """
  Processa a redação completa e gera todos os resultados necessários.
  
  Args:
      redacao_texto: Texto da redação
      tema_redacao: Tema da redação
      cohmetrix_results: Resultados da análise Coh-Metrix
      user_id: ID do usuário
      
  Returns:
      Dict contendo todos os resultados da análise
  """
  logger.info("Iniciando processamento da redação")
  logger.info(f"Estados presentes: {st.session_state.keys()}")

  resultados = {
      'analises_detalhadas': {},
      'notas': {},
      'nota_total': 0,
      'erros_especificos': {},
      'justificativas': {},
      'total_erros_por_competencia': {},
      'sugestoes_estilo': {},
      'texto_original': redacao_texto
  }
  
  # Processar cada competência
  for comp, descricao in competencies.items():
      # Obter funções de análise e atribuição de nota para a competência
      analise_func = globals()[f"analisar_{comp}"]
      atribuir_nota_func = globals()[f"atribuir_nota_{comp}"]
      
      # Realizar análise da competência
      resultado_analise = analise_func(redacao_texto, tema_redacao, cohmetrix_results)
      
      # Garantir que erros existam, mesmo que vazio
      erros_revisados = resultado_analise.get('erros', [])
      
      # Atribuir nota baseado na análise completa e erros
      resultado_nota = atribuir_nota_func(resultado_analise['analise'], erros_revisados)
      nota = resultado_nota['nota']
      justificativa = resultado_nota['justificativa']
      
      # Preencher resultados para esta competência
      resultados['analises_detalhadas'][comp] = resultado_analise['analise']
      resultados['notas'][comp] = nota
      resultados['justificativas'][comp] = justificativa
      resultados['erros_especificos'][comp] = erros_revisados
      resultados['total_erros_por_competencia'][comp] = len(erros_revisados)
      
      # Incluir sugestões de estilo se existirem
      if 'sugestoes_estilo' in resultado_analise:
          resultados['sugestoes_estilo'][comp] = resultado_analise['sugestoes_estilo']

  # Calcular nota total
  resultados['nota_total'] = sum(resultados['notas'].values())
  
  # Salvar no session_state
  st.session_state.analise_realizada = True
  st.session_state.resultados = resultados
  st.session_state.redacao_texto = redacao_texto
  st.session_state.tema_redacao = tema_redacao
  st.session_state.erros_especificos_todas_competencias = resultados['erros_especificos']
  st.session_state.notas_atualizadas = resultados['notas'].copy()
  
  # Adicionar timestamp da análise em formato ISO
  try:
      st.session_state.ultima_analise_timestamp = datetime.now().isoformat()
  except Exception as e:
      logger.error(f"Erro ao salvar timestamp: {e}")
      st.session_state.ultima_analise_timestamp = None
  
  # Salvar no Elasticsearch
  try:
      save_redacao_es(user_id, redacao_texto, tema_redacao, resultados['notas'], resultados['analises_detalhadas'])
  except Exception as e:
      logger.error(f"Erro ao salvar no Elasticsearch: {str(e)}")
  
  # Salvar no Supabase
  try:
      save_redacao(
          user_id,
          redacao_texto,
          tema_redacao,
          resultados['notas'],
          resultados['analises_detalhadas']
      )
  except Exception as e:
      logger.error(f"Erro ao salvar no Supabase: {str(e)}")
  
  logger.info("Processamento concluído. Resultados gerados.")
  logger.info(f"Estados após processamento: {st.session_state.keys()}")
  
  return resultados

def analisar_competency1(redacao_texto: str, tema_redacao: str, cohmetrix_results: Dict[str, int]) -> Dict[str, Any]:
    """
    Análise da Competência 1: Domínio da Norma Culta.
    Identifica apenas erros reais que devem penalizar a nota, separando sugestões estilísticas.
    
    Args:
        redacao_texto: Texto da redação
        tema_redacao: Tema da redação
        cohmetrix_results: Métricas textuais do Coh-Metrix
        
    Returns:
        Dict contendo análise, erros, sugestões e total de erros
    """
    
    MODELO_COMP1 = "ft:gpt-4o-2024-08-06:personal:competencia-1:AHDQQucG"
    
    criterios = {
        "ortografia": """
        Analise o texto linha por linha quanto à ortografia, identificando APENAS ERROS REAIS em:
        1. Palavras escritas incorretamente
        2. Problemas de acentuação
        3. Uso incorreto de maiúsculas/minúsculas
        4. Grafia de estrangeirismos
        5. Abreviações inadequadas
        
        NÃO inclua sugestões de melhoria ou preferências estilísticas.
        Inclua apenas desvios claros da norma culta.
        
        Texto para análise: {redacao_texto}
        
        Para cada ERRO REAL encontrado, forneça:
        ERRO
        Descrição: [Descrição objetiva do erro ortográfico]
        Trecho: "[Trecho exato do texto]"
        Explicação: [Explicação técnica do erro]
        Sugestão: [Correção necessária]
        FIM_ERRO
        """,
        
        "pontuacao": """
        Analise o texto linha por linha quanto à pontuação, identificando APENAS ERROS REAIS em:
        1. Uso incorreto de vírgulas em:
           - Enumerações
           - Orações coordenadas
           - Orações subordinadas
           - Apostos e vocativos
           - Adjuntos adverbiais deslocados
        2. Uso inadequado de ponto e vírgula
        3. Uso incorreto de dois pontos
        4. Problemas com pontos finais
        5. Uso inadequado de reticências
        6. Problemas com travessões e parênteses
        
        NÃO inclua sugestões de melhoria ou pontuação opcional.
        Inclua apenas desvios claros das regras de pontuação.
        
        Texto para análise: {redacao_texto}
        
        Para cada ERRO REAL encontrado, forneça:
        ERRO
        Descrição: [Descrição objetiva do erro de pontuação]
        Trecho: "[Trecho exato do texto]"
        Explicação: [Explicação técnica do erro]
        Sugestão: [Correção necessária]
        FIM_ERRO
        """,
       
       "concordancia": """
        Analise o texto linha por linha quanto à concordância, identificando APENAS ERROS REAIS em:
        1. Concordância verbal
           - Sujeito e verbo
           - Casos especiais (coletivos, expressões partitivas)
        2. Concordância nominal
           - Substantivo e adjetivo
           - Casos especiais (é necessário, é proibido)
        3. Concordância ideológica
        4. Silepse (de gênero, número e pessoa)
        
        NÃO inclua sugestões de melhoria ou preferências de concordância.
        Inclua apenas desvios claros das regras de concordância.
        
        Texto para análise: {redacao_texto}
        
        Para cada ERRO REAL encontrado, forneça:
        ERRO
        Descrição: [Descrição objetiva do erro de concordância]
        Trecho: "[Trecho exato do texto]"
        Explicação: [Explicação técnica do erro]
        Sugestão: [Correção necessária]
        FIM_ERRO
        """,
        
        "regencia": """
        Analise o texto linha por linha quanto à regência, identificando APENAS ERROS REAIS em:
        1. Regência verbal
           - Uso inadequado de preposições com verbos
           - Ausência de preposição necessária
        2. Regência nominal
           - Uso inadequado de preposições com nomes
        3. Uso da crase: Verifique CUIDADOSAMENTE se há:
           - Junção de preposição 'a' com artigo definido feminino 'a'
           - Palavra feminina usada em sentido definido
           - Locuções adverbiais femininas
           
        IMPORTANTE: Analise cada caso considerando:
        - O contexto completo da frase
        - A função sintática das palavras
        - O sentido pretendido (definido/indefinido)
        - A regência dos verbos e nomes envolvidos
        
        NÃO marque como erro casos onde:
        - Não há artigo definido feminino
        - A palavra está sendo usada em sentido indefinido
        - Há apenas preposição 'a' sem artigo
        
        Texto para análise: {redacao_texto}
        
        Para cada ERRO REAL encontrado, forneça:
        ERRO
        Descrição: [Descrição objetiva do erro de regência]
        Trecho: "[Trecho exato do texto]"
        Explicação: [Explicação técnica DETALHADA do erro, incluindo análise sintática]
        Sugestão: [Correção necessária com justificativa]
        FIM_ERRO
        """
    }
    
    erros_por_criterio = {}
    for criterio, prompt in criterios.items():
        prompt_formatado = prompt.format(redacao_texto=redacao_texto)
        resposta = client.chat.completions.create(
            model=MODELO_COMP1,
            messages=[{"role": "user", "content": prompt_formatado}],
            temperature=0.3
        )
        erros_por_criterio[criterio] = extrair_erros_do_resultado(resposta.choices[0].message.content)
    
    todos_erros = []
    for erros in erros_por_criterio.values():
        todos_erros.extend(erros)
   
   # Separar erros reais de sugestões estilísticas
    erros_reais = []
    sugestoes_estilo = []
    
    palavras_chave_sugestao = [
        "pode ser melhorada",
        "poderia ser",
        "considerar",
        "sugerimos",
        "recomendamos",
        "ficaria melhor",
        "seria preferível",
        "opcionalmente",
        "para aprimorar",
        "para enriquecer",
        "estilo",
        "clareza",
        "mais elegante",
        "sugestão de melhoria",
        "alternativa",
        "opcional"
    ]
    
    for erro in todos_erros:
        eh_sugestao = False
        explicacao = erro.get('explicação', '').lower()
        sugestao = erro.get('sugestão', '').lower()
        
        # Verificar se é uma sugestão
        for palavra in palavras_chave_sugestao:
            if (palavra in explicacao or palavra in sugestao):
                eh_sugestao = True
                break
        
        # Verificar outros indicadores de sugestão vs erro
        if (eh_sugestao or 
            any(palavra in explicacao for palavra in ['pode', 'poderia', 'opcional', 'talvez']) or
            'recomend' in explicacao or
            'suggestion' in explicacao):
            sugestoes_estilo.append(erro)
        else:
            # Validação adicional para erros de crase
            if "crase" in erro.get('descrição', '').lower():
                explicacao = erro.get('explicação', '').lower()
                # Só considera erro se houver justificativa técnica clara
                if any(termo in explicacao for termo in ['artigo definido', 'sentido definido', 'locução']) and \
                   any(termo in explicacao for termo in ['regência', 'preposição', 'artigo feminino']):
                    erros_reais.append(erro)
            else:
                erros_reais.append(erro)
    
    # Revisão final dos erros reais
    erros_revisados = revisar_erros_competency1(erros_reais, redacao_texto)
    
    # Gerar análise final apenas com erros confirmados
    prompt_analise = f"""
    Com base nos seguintes ERROS CONFIRMADOS no texto (excluindo sugestões de melhoria estilística),
    gere uma análise detalhada da Competência 1 (Domínio da Norma Culta):
    
    Total de erros confirmados: {len(erros_revisados)}
    
    Detalhamento dos erros confirmados:
    {json.dumps(erros_revisados, indent=2)}
    
    Observação: Analisar apenas os erros reais que prejudicam a nota, ignorando sugestões de melhoria.
    
    Forneça uma análise que:
    1. Avalie o domínio geral da norma culta considerando apenas erros confirmados
    2. Destaque os tipos de erros mais frequentes e sua gravidade
    3. Analise o impacto dos erros na compreensão do texto
    4. Avalie a consistência no uso da norma culta
    5. Forneça uma visão geral da qualidade técnica do texto
    
    Formato da resposta:
    Análise Geral: [Sua análise aqui]
    Erros Principais: [Lista dos erros mais relevantes]
    Impacto na Compreensão: [Análise do impacto dos erros]
    Consistência: [Avaliação da consistência no uso da norma]
    Conclusão: [Visão geral da qualidade técnica]
    """
    
    resposta_analise = client.chat.completions.create(
        model=MODELO_COMP1,
        messages=[{"role": "user", "content": prompt_analise}],
        temperature=0.3
    )
    analise_geral = resposta_analise.choices[0].message.content
    
    return {
        'analise': analise_geral,
        'erros': erros_revisados,
        'sugestoes_estilo': sugestoes_estilo,
        'total_erros': len(erros_revisados)
    }

def revisar_erros_competency1(erros_identificados: List[Dict], redacao_texto: str) -> List[Dict]:
    """
    Revisa os erros identificados na Competência 1 usando análise contextual aprofundada.
    
    Args:
        erros_identificados: Lista de erros identificados inicialmente
        redacao_texto: Texto completo da redação para análise contextual
        
    Returns:
        Lista de erros validados e revisados
    """
    MODELO_REVISAO_COMP1 = "ft:gpt-4o-2024-08-06:personal:competencia-1:AHDQQucG"
    erros_revisados = []
    
    for erro in erros_identificados:
        # Extrair contexto expandido do erro
        trecho = erro.get('trecho', '')
        inicio_trecho = redacao_texto.find(trecho)
        if inicio_trecho != -1:
            # Pegar até 100 caracteres antes e depois para contexto
            inicio_contexto = max(0, inicio_trecho - 100)
            fim_contexto = min(len(redacao_texto), inicio_trecho + len(trecho) + 100)
            contexto_expandido = redacao_texto[inicio_contexto:fim_contexto]
        else:
            contexto_expandido = trecho
            
        prompt_revisao = f"""
        Revise rigorosamente o seguinte erro identificado na Competência 1 (Domínio da Norma Culta).
        
        Erro original:
        {json.dumps(erro, indent=2)}

        Contexto expandido do erro:
        "{contexto_expandido}"

        Texto completo para referência:
        {redacao_texto}

        Analise cuidadosamente:
        1. CONTEXTO SINTÁTICO:
           - Estrutura completa da frase
           - Função sintática das palavras
           - Relações de dependência
           
        2. REGRAS GRAMATICAIS:
           - Regras específicas aplicáveis
           - Exceções relevantes
           - Casos especiais
           
        3. IMPACTO NO SENTIDO:
           - Se o suposto erro realmente compromete a compreensão
           - Se há ambiguidade ou prejuízo ao sentido
           - Se é um desvio real ou variação aceitável
           
        4. ADEQUAÇÃO AO ENEM:
           - Critérios específicos da prova
           - Impacto na avaliação
           - Relevância do erro

        Para casos de crase, VERIFIQUE ESPECIFICAMENTE:
        - Se há realmente junção de preposição 'a' com artigo definido feminino
        - Se a palavra está sendo usada em sentido definido
        - Se há regência verbal/nominal exigindo preposição
        - O contexto completo da construção

        Formato da resposta:
        REVISAO
        Erro Confirmado: [Sim/Não]
        Análise Sintática: [Análise detalhada da estrutura sintática]
        Regra Aplicável: [Citação da regra gramatical específica]
        Explicação Revisada: [Explicação técnica detalhada]
        Sugestão Revisada: [Correção com justificativa]
        Considerações ENEM: [Relevância para a avaliação]
        FIM_REVISAO
        """
        
        try:
            resposta_revisao = client.chat.completions.create(
                model=MODELO_REVISAO_COMP1,
                messages=[{"role": "user", "content": prompt_revisao}],
                temperature=0.2
            )
            
            revisao = extrair_revisao_do_resultado(resposta_revisao.choices[0].message.content)
            
            # Validação rigorosa da revisão
            if (revisao['Erro Confirmado'] == 'Sim' and
                'Análise Sintática' in revisao and
                'Regra Aplicável' in revisao and
                len(revisao.get('Explicação Revisada', '')) > 50):  # Garantir explicação substancial
                
                erro_revisado = erro.copy()
                erro_revisado.update({
                    'análise_sintática': revisao['Análise Sintática'],
                    'regra_aplicável': revisao['Regra Aplicável'],
                    'explicação': revisao['Explicação Revisada'],
                    'sugestão': revisao['Sugestão Revisada'],
                    'considerações_enem': revisao['Considerações ENEM'],
                    'contexto_expandido': contexto_expandido
                })
                
                # Validação adicional para erros de crase
                if "crase" in erro.get('descrição', '').lower():
                    explicacao = revisao['Explicação Revisada'].lower()
                    analise = revisao['Análise Sintática'].lower()
                    
                    # Só aceita se houver análise técnica completa
                    if ('artigo definido' in explicacao and
                        'preposição' in explicacao and
                        any(termo in analise for termo in ['função sintática', 'regência', 'complemento'])):
                        erros_revisados.append(erro_revisado)
                else:
                    erros_revisados.append(erro_revisado)
                    
        except Exception as e:
            logging.error(f"Erro ao revisar: {str(e)}")
            continue
    
    return erros_revisados

def extrair_revisao_do_resultado(texto):
    revisao = {}
    linhas = texto.split('\n')
    for linha in linhas:
        if ':' in linha:
            chave, valor = linha.split(':', 1)
            revisao[chave.strip()] = valor.strip()
    return revisao


def analisar_competency2(redacao_texto: str, tema_redacao: str, cohmetrix_results: Dict[str, int]) -> Dict[str, Any]:
    """Análise da Competência 2: Compreensão do Tema"""
    prompt_analise = f"""
    Analise a compreensão do tema na seguinte redação, considerando:
    1. Texto da redação: {redacao_texto}
    2. Tema proposto: {tema_redacao}
    3. Métricas textuais:
       - Número de palavras: {cohmetrix_results["Word Count"]}
       - Número de sentenças: {cohmetrix_results["Sentence Count"]}
       - Palavras únicas: {cohmetrix_results["Unique Words"]}
       - Diversidade lexical: {cohmetrix_results["Lexical Diversity"]}
    Forneça uma análise detalhada, incluindo:
    1. Avaliação do domínio do tema proposto.
    2. Análise da presença das palavras principais do tema ou seus sinônimos em cada parágrafo.
    3. Avaliação da argumentação e uso de repertório sociocultural.
    4. Análise da clareza do ponto de vista adotado.
    5. Avaliação do vínculo entre o repertório e a discussão proposta.
    6. Verificação de cópia de trechos dos textos motivadores.
    7. Análise da citação de fontes do repertório utilizado.
    
    Para cada ponto analisado que represente um erro ou área de melhoria, forneça um exemplo específico do texto, no seguinte formato:
    ERRO
    Trecho: "[Trecho exato do texto]"
    Explicação: [Explicação detalhada]
    Sugestão: [Sugestão de melhoria]
    FIM_ERRO

    Se não houver erros significativos, indique isso claramente na análise.

    Formato da resposta:
    Domínio do Tema: [Sua análise aqui]
    Uso de Palavras-chave: [Sua análise aqui]
    Argumentação e Repertório: [Sua análise aqui]
    Clareza do Ponto de Vista: [Sua análise aqui]
    Vínculo Repertório-Discussão: [Sua análise aqui]
    Originalidade: [Sua análise aqui]
    Citação de Fontes: [Sua análise aqui]
    """
    docs_relevantes = retrieve_relevant_docs("Compreensão do Tema ENEM")
    analise_geral = generate_rag_response(prompt_analise, docs_relevantes, "competency2")
    
    # Remover blocos de ERRO do texto da análise
    analise_limpa = re.sub(r'ERRO\n.*?FIM_ERRO', '', analise_geral, flags=re.DOTALL)
    
    erros_identificados = extrair_erros_do_resultado(analise_geral)
    erros_revisados = revisar_erros_competency2(erros_identificados, redacao_texto)

    return {
        'analise': analise_limpa,
        'erros': erros_revisados
    }
def extrair_erros_do_resultado(resultado: str) -> List[Dict[str, str]]:
    erros = []
    padrao_erro = re.compile(r'ERRO\n(.*?)\nFIM_ERRO', re.DOTALL)
    matches = padrao_erro.findall(resultado)
    
    for match in matches:
        erro = {}
        for linha in match.split('\n'):
            if ':' in linha:
                chave, valor = linha.split(':', 1)
                chave = chave.strip().lower()
                valor = valor.strip()
                if chave == 'trecho':
                    valor = valor.strip('"')
                erro[chave] = valor
        if 'descrição' in erro and 'trecho' in erro:
            erros.append(erro)
    
    return erros


def analisar_competency3(redacao_texto: str, tema_redacao: str, cohmetrix_results: Dict[str, int]) -> Dict[str, Any]:
    """Análise da Competência 3: Seleção e Organização das Informações"""
    prompt_analise = f"""
    Analise a seleção e organização das informações na seguinte redação, considerando:
    1. Texto da redação: {redacao_texto}
    2. Tema: {tema_redacao}
    3. Métricas textuais:
       - Número de parágrafos: {cohmetrix_results["Paragraph Count"]}
       - Média de sentenças por parágrafo: {cohmetrix_results["Sentences per Paragraph"]}
       - Uso de conectivos: {cohmetrix_results["Connectives"]}
       - Frases nominais: {cohmetrix_results["Noun Phrases"]}
       - Frases verbais: {cohmetrix_results["Verb Phrases"]}

    Forneça uma análise detalhada, incluindo:
    1. Avaliação da progressão das ideias e seleção de argumentos.
    2. Análise da organização das informações e fatos relacionados ao tema.
    3. Comentários sobre a defesa do ponto de vista e consistência argumentativa.
    4. Avaliação da autoria e originalidade das informações apresentadas.
    5. Análise do encadeamento das ideias entre parágrafos.
    6. Verificação de repetições desnecessárias ou saltos temáticos.
    7. Avaliação da estrutura de cada parágrafo (argumento, justificativa, repertório, justificativa, frase de finalização).

    Para cada ponto analisado que represente um erro ou área de melhoria, forneça um exemplo específico do texto, no seguinte formato:
    ERRO
    Trecho: "[Trecho exato do texto]"
    Explicação: [Explicação detalhada]
    Sugestão: [Sugestão de melhoria]
    FIM_ERRO

    Se não houver erros significativos, indique isso claramente na análise.

    Formato da resposta:
    Progressão de Ideias: [Sua análise aqui]
    Organização de Informações: [Sua análise aqui]
    Defesa do Ponto de Vista: [Sua análise aqui]
    Autoria e Originalidade: [Sua análise aqui]
    Encadeamento entre Parágrafos: [Sua análise aqui]
    Estrutura dos Parágrafos: [Sua análise aqui]
    """
    docs_relevantes = retrieve_relevant_docs("Seleção e Organização das Informações ENEM")
    analise_geral = generate_rag_response(prompt_analise, docs_relevantes, "competency3")

    # Remover blocos de ERRO do texto da análise
    analise_limpa = re.sub(r'ERRO\n.*?FIM_ERRO', '', analise_geral, flags=re.DOTALL)

    erros_identificados = extrair_erros_do_resultado(analise_geral)
    erros_revisados = revisar_erros_competency3(erros_identificados, redacao_texto)

    return {
        'analise': analise_limpa,
        'erros': erros_revisados
    }


def analisar_competency4(redacao_texto: str, tema_redacao: str, cohmetrix_results: Dict[str, int]) -> Dict[str, Any]:
    """Análise da Competência 4: Conhecimento dos Mecanismos Linguísticos"""
    prompt_analise = f"""
    Analise o conhecimento dos mecanismos linguísticos na seguinte redação, considerando:
    1. Texto da redação: {redacao_texto}
    2. Tema: {tema_redacao}
    3. Métricas textuais:
       - Uso de conectivos: {cohmetrix_results["Connectives"]}
       - Média de palavras por sentença: {cohmetrix_results["Words per Sentence"]}
       - Frases nominais: {cohmetrix_results["Noun Phrases"]}
       - Frases verbais: {cohmetrix_results["Verb Phrases"]}

    Forneça uma análise detalhada, incluindo:
    1. Avaliação do uso de conectivos no início de cada período.
    2. Análise da articulação entre as partes do texto.
    3. Avaliação do repertório de recursos coesivos.
    4. Análise do uso de referenciação (pronomes, sinônimos, advérbios).
    5. Avaliação das transições entre ideias (causa/consequência, comparação, conclusão).
    6. Análise da organização de períodos complexos.
    7. Verificação da repetição de conectivos ao longo do texto.

    Para cada ponto analisado que represente um erro ou área de melhoria, forneça um exemplo específico do texto, no seguinte formato:
    ERRO
    Trecho: "[Trecho exato do texto]"
    Explicação: [Explicação detalhada]
    Sugestão: [Sugestão de melhoria]
    FIM_ERRO

    Se não houver erros significativos, indique isso claramente na análise.

    Formato da resposta:
    Uso de Conectivos: [Sua análise aqui]
    Articulação Textual: [Sua análise aqui]
    Recursos Coesivos: [Sua análise aqui]
    Referenciação: [Sua análise aqui]
    Transições de Ideias: [Sua análise aqui]
    Estrutura de Períodos: [Sua análise aqui]
    """
    docs_relevantes = retrieve_relevant_docs("Conhecimento dos Mecanismos Linguísticos ENEM")
    analise_geral = generate_rag_response(prompt_analise, docs_relevantes, "competency4")

    # Remover blocos de ERRO do texto da análise
    analise_limpa = re.sub(r'ERRO\n.*?FIM_ERRO', '', analise_geral, flags=re.DOTALL)

    erros_identificados = extrair_erros_do_resultado(analise_geral)
    erros_revisados = revisar_erros_competency4(erros_identificados, redacao_texto)

    return {
        'analise': analise_limpa,
        'erros': erros_revisados
    }

def analisar_competency5(redacao_texto: str, tema_redacao: str, cohmetrix_results: Dict[str, int]) -> Dict[str, Any]:
    """Análise da Competência 5: Proposta de Intervenção"""
    prompt_analise = f"""
    Analise a proposta de intervenção na seguinte redação, considerando:
    1. Texto da redação: {redacao_texto}
    2. Tema: {tema_redacao}
    3. Métricas textuais:
       - Número de sentenças: {cohmetrix_results["Sentence Count"]}
       - Número de palavras: {cohmetrix_results["Word Count"]}
       - Número de parágrafos: {cohmetrix_results["Paragraph Count"]}

    Forneça uma análise detalhada, incluindo:
    1. Avaliação da presença dos cinco elementos obrigatórios: agente, ação, modo/meio, detalhamento e finalidade.
    2. Análise do nível de detalhamento e articulação da proposta com a discussão do texto.
    3. Avaliação da viabilidade e respeito aos direitos humanos na proposta.
    4. Verificação da retomada do contexto inicial (se houver).
    5. Análise da coerência entre a proposta e o tema discutido.

    Para cada ponto que represente um erro ou área de melhoria, forneça um exemplo específico do texto no seguinte formato:
    ERRO
    Trecho: "[Trecho exato do texto]"
    Explicação: [Explicação detalhada]
    Sugestão: [Sugestão de melhoria]
    FIM_ERRO

    Se não houver erros significativos, indique isso claramente na análise.

    Formato da resposta:
    Elementos da Proposta: [Sua análise aqui]
    Detalhamento e Articulação: [Sua análise aqui]
    Viabilidade e Direitos Humanos: [Sua análise aqui]
    Retomada do Contexto: [Sua análise aqui]
    Coerência com o Tema: [Sua análise aqui]
    """
    docs_relevantes = retrieve_relevant_docs("Proposta de Intervenção ENEM")
    analise_geral = generate_rag_response(prompt_analise, docs_relevantes, "competency5")

    # Remover blocos de ERRO do texto da análise
    analise_limpa = re.sub(r'ERRO\n.*?FIM_ERRO', '', analise_geral, flags=re.DOTALL)

    erros_identificados = extrair_erros_do_resultado(analise_geral)
    erros_revisados = revisar_erros_competency5(erros_identificados, redacao_texto)

    return {
        'analise': analise_limpa,
        'erros': erros_revisados
    }
def revisar_erros_competency2(erros_identificados, redacao_texto):
    """Revisa os erros identificados na Competência 2 usando um modelo FT e base RAG do ENEM"""
    
    MODELO_REVISAO_COMP2 = "ft:gpt-4o-2024-08-06:personal:competencia-2:AHDT84HO"
    
    return revisar_erros_generico(erros_identificados, redacao_texto, MODELO_REVISAO_COMP2, "Compreensão do Tema")

def revisar_erros_competency3(erros_identificados, redacao_texto):
    """Revisa os erros identificados na Competência 3 usando um modelo FT e base RAG do ENEM"""
    
    MODELO_REVISAO_COMP3 = "ft:gpt-4o-2024-08-06:personal:competencia-3:AHDUfZRb"
    
    return revisar_erros_generico(erros_identificados, redacao_texto, MODELO_REVISAO_COMP3, "Seleção e Organização das Informações")

def revisar_erros_competency4(erros_identificados, redacao_texto):
    """Revisa os erros identificados na Competência 4 usando um modelo FT e base RAG do ENEM"""
    
    MODELO_REVISAO_COMP4 = "ft:gpt-4o-2024-08-06:personal:competencia-4:AHDXewU3"
    
    return revisar_erros_generico(erros_identificados, redacao_texto, MODELO_REVISAO_COMP4, "Conhecimento dos Mecanismos Linguísticos")

def revisar_erros_competency5(erros_identificados, redacao_texto):
    """Revisa os erros identificados na Competência 5 usando um modelo FT e base RAG do ENEM"""
    
    MODELO_REVISAO_COMP5 = "ft:gpt-4o-2024-08-06:personal:competencia-5:AHGVPnJG"
    
    return revisar_erros_generico(erros_identificados, redacao_texto, MODELO_REVISAO_COMP5, "Proposta de Intervenção")


def revisar_erros_generico(erros_identificados, redacao_texto, modelo_revisao, nome_competencia):
    """Função genérica para revisar erros de qualquer competência"""
    
    erros_revisados = []
    
    for erro in erros_identificados:
        prompt_revisao = f"""
        Revise o seguinte erro identificado na Competência {nome_competencia} 
        de acordo com os critérios específicos do ENEM:

        Erro original:
        {json.dumps(erro, indent=2)}

        Texto da redação:
        {redacao_texto}

        Com base nos critérios do ENEM e na base de conhecimento RAG, determine:
        1. Se o erro está corretamente identificado
        2. Se a explicação e sugestão estão adequadas aos padrões do ENEM
        3. Se há alguma consideração adicional relevante para o contexto do ENEM

        Formato da resposta:
        REVISAO
        Erro Confirmado: [Sim/Não]
        Explicação Revisada: [Nova explicação, se necessário]
        Sugestão Revisada: [Nova sugestão, se necessário]
        Considerações ENEM: [Observações específicas sobre o erro no contexto do ENEM]
        FIM_REVISAO
        """
        
        resposta_revisao = client.chat.completions.create(
            model=modelo_revisao,
            messages=[{"role": "user", "content": prompt_revisao}],
            temperature=0.2
        )
        
        revisao = extrair_revisao_do_resultado(resposta_revisao.choices[0].message.content)
        
        if revisao['Erro Confirmado'] == 'Sim':
            erro_revisado = erro.copy()
            if 'Explicação Revisada' in revisao:
                erro_revisado['explicação'] = revisao['Explicação Revisada']
            if 'Sugestão Revisada' in revisao:
                erro_revisado['sugestão'] = revisao['Sugestão Revisada']
            erro_revisado['considerações_enem'] = revisao['Considerações ENEM']
            erros_revisados.append(erro_revisado)
    
    return erros_revisados

def atribuir_nota_competency1(analise: str, erros: List[Dict[str, str]]) -> Dict[str, Any]:
   """
   Atribui nota à Competência 1 com base na análise detalhada e erros identificados.
   
   Args:
       analise: String contendo a análise detalhada do texto
       erros: Lista de dicionários contendo os erros identificados
       
   Returns:
       Dict contendo a nota atribuída (0-200) e sua justificativa
   """
   # Contar erros por categoria
   contagem_erros = {
       'sintaxe': 0,
       'ortografia': 0, 
       'concordancia': 0,
       'pontuacao': 0,
       'crase': 0,
       'registro': 0
   }
   
   for erro in erros:
       desc = erro.get('explicacao', '').lower()
       if 'sintax' in desc or 'estrutura' in desc:
           contagem_erros['sintaxe'] += 1
       if 'ortograf' in desc or 'accent' in desc or 'escrita' in desc:
           contagem_erros['ortografia'] += 1
       if 'concord' in desc or 'verbal' in desc or 'nominal' in desc:
           contagem_erros['concordancia'] += 1
       if 'pontu' in desc:
           contagem_erros['pontuacao'] += 1
       if 'crase' in desc:
           contagem_erros['crase'] += 1
       if 'coloquial' in desc or 'registro' in desc or 'informal' in desc:
           contagem_erros['registro'] += 1

   # Formatar erros para apresentação
   erros_formatados = ""
   for erro in erros:
       erros_formatados += f"""
       Erro encontrado:
       Trecho: "{erro.get('trecho', '')}"
       Explicação: {erro.get('explicacao', '')}"
       Sugestão: {erro.get('sugestao', '')}
       """

   # Determinar nota base pelos critérios objetivos
   total_erros = sum(contagem_erros.values())
   if (total_erros <= 3 and 
       contagem_erros['sintaxe'] <= 1 and
       contagem_erros['registro'] == 0 and
       contagem_erros['ortografia'] <= 1):
       nota_base = 200
   elif (total_erros <= 5 and 
         contagem_erros['sintaxe'] <= 2 and
         contagem_erros['registro'] <= 1):
       nota_base = 160
   elif (total_erros <= 8 and 
         contagem_erros['sintaxe'] <= 3):
       nota_base = 120
   elif total_erros <= 12:
       nota_base = 80
   elif total_erros <= 15:
       nota_base = 40
   else:
       nota_base = 0

   # Construir prompt para validação da nota
   prompt_nota = f"""
   Com base na seguinte análise da Competência 1 (Domínio da Norma Culta) e na contagem de erros identificados,
   confirme se a nota {nota_base} está adequada.
   
   ANÁLISE DETALHADA:
   {analise}
   
   CONTAGEM DE ERROS:
   - Erros de sintaxe/estrutura: {contagem_erros['sintaxe']}
   - Erros de ortografia/acentuação: {contagem_erros['ortografia']}
   - Erros de concordância: {contagem_erros['concordancia']}
   - Erros de pontuação: {contagem_erros['pontuacao']}
   - Erros de crase: {contagem_erros['crase']}
   - Desvios de registro formal: {contagem_erros['registro']}
   Total de erros: {total_erros}
   
   ERROS ESPECÍFICOS:
   {erros_formatados}
   
   Critérios para cada nota:
   
   200 pontos:
   - No máximo uma falha de estrutura sintática
   - No máximo dois desvios gramaticais
   - Nenhum uso de linguagem informal/coloquial
   - No máximo um erro ortográfico
   - Coerência e coesão impecáveis
   - Sem repetição de erros
   
   160 pontos:
   - Até três desvios gramaticais que não comprometem a compreensão
   - Poucos erros de pontuação/acentuação
   - No máximo três erros ortográficos
   - Bom domínio geral da norma culta
   
   120 pontos:
   - Até cinco desvios gramaticais
   - Domínio mediano da norma culta
   - Alguns problemas de coesão pontuais
   - Erros não sistemáticos
   
   80 pontos:
   - Estrutura sintática deficitária
   - Erros frequentes de concordância
   - Uso ocasional de registro inadequado
   - Muitos erros de pontuação/ortografia
   
   40 pontos:
   - Domínio precário da norma culta
   - Diversos desvios gramaticais frequentes
   - Problemas graves de coesão
   - Registro frequentemente inadequado
   
   0 pontos:
   - Desconhecimento total da norma culta
   - Erros graves e sistemáticos
   - Texto incompreensível
   
   Com base nesses critérios e na análise apresentada, forneça:
   1. Confirmação ou ajuste da nota base {nota_base}
   2. Justificativa detalhada relacionando os erros encontrados com os critérios
   
   Formato da resposta:
   Nota: [NOTA FINAL]
   Justificativa: [Justificativa detalhada da nota, explicando como os erros e acertos se relacionam com os critérios]
   """
   
   # Gerar resposta usando RAG
   docs_relevantes = retrieve_relevant_docs("Critérios de Avaliação Competência 1 ENEM")
   resposta_nota = generate_rag_response(prompt_nota, docs_relevantes, "competency1_nota")
   
   # Extrair nota e justificativa
   resultado = extrair_nota_e_justificativa(resposta_nota)
   
   # Validar se a nota está nos valores permitidos
   if resultado['nota'] not in [0, 40, 80, 120, 160, 200]:
       resultado['nota'] = nota_base
       resultado['justificativa'] += "\nNota ajustada para o valor válido mais próximo."
   
   # Validar discrepância com nota base
   if abs(resultado['nota'] - nota_base) > 40:
       resultado['nota'] = min(nota_base, resultado['nota'])
       resultado['justificativa'] += "\nNota ajustada devido à quantidade e gravidade dos erros identificados."
   
   return resultado

    
def atribuir_nota_competency2(analise: str, erros: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_nota = f"""
    Com base na seguinte análise da Competência 2 (Compreensão do Tema) do ENEM, atribua uma nota de 0 a 200 em intervalos de 40 pontos (0, 40, 80, 120, 160 ou 200).

    Análise:
    {analise}

    Considere cuidadosamente os seguintes critérios para atribuir a nota:

    Nota 200:
    - Excelente domínio do tema proposto.
    - Citação das palavras principais do tema ou sinônimos em cada parágrafo.
    - Argumentação consistente com repertório sociocultural produtivo.
    - Uso de exemplos históricos, frases, músicas, textos, autores famosos, filósofos, estudos, artigos ou publicações como repertório.
    - Excelente domínio do texto dissertativo-argumentativo, incluindo proposição, argumentação e conclusão.
    - Não copia trechos dos textos motivadores e demonstra clareza no ponto de vista adotado.
    - Estabelece vínculo de ideias entre a referência ao repertório e a discussão proposta.
    - Cita a fonte do repertório (autor, obra, data de criação, etc.).
    - Inclui pelo menos um repertório no segundo e terceiro parágrafo.

    Nota 160:
    - Bom desenvolvimento do tema com argumentação consistente, mas sem repertório sociocultural tão produtivo.
    - Completa as 3 partes do texto dissertativo-argumentativo (nenhuma delas é embrionária).
    - Bom domínio do texto dissertativo-argumentativo, com proposição, argumentação e conclusão claras, mas sem aprofundamento.
    - Utiliza informações pertinentes, mas sem extrapolar significativamente sua justificativa.

    Nota 120:
    - Abordagem completa do tema, com as 3 partes do texto dissertativo-argumentativo (podendo 1 delas ser embrionária).
    - Repertório baseado nos textos motivadores e/ou repertório não legitimado e/ou repertório legitimado, mas não pertinente ao tema.
    - Desenvolvimento do tema de forma previsível, com argumentação mediana, sem grandes inovações.
    - Domínio mediano do texto dissertativo-argumentativo, com proposição, argumentação e conclusão, mas de forma superficial.

    Nota 80:
    - Abordagem completa do tema, mas com problemas relacionados ao tipo textual e presença de muitos trechos de cópia sem aspas.
    - Domínio insuficiente do texto dissertativo-argumentativo, faltando a estrutura completa de proposição, argumentação e conclusão.
    - Não desenvolve um ponto de vista claro e não consegue conectar as ideias argumentativas adequadamente.
    - Duas partes embrionárias ou com conclusão finalizada por frase incompleta.

    Nota 40:
    - Tangencia o tema, sem abordar diretamente o ponto central proposto.
    - Domínio precário do texto dissertativo-argumentativo, com traços de outros tipos textuais.
    - Não constrói uma argumentação clara e objetiva, resultando em confusão ou desvio do gênero textual.

    Nota 0:
    - Fuga completa do tema proposto, abordando um assunto irrelevante ou não relacionado.
    - Não atende à estrutura dissertativo-argumentativa, sendo classificado como outro gênero textual.
    - Não apresenta proposição, argumentação e conclusão, ou o texto é anulado por não atender aos critérios básicos de desenvolvimento textual.

    Forneça a nota e uma justificativa detalhada, relacionando diretamente com a análise fornecida. Certifique-se de que a justificativa esteja completamente alinhada com a nota atribuída e os critérios específicos.

    Formato da resposta:
    Nota: [NOTA ATRIBUÍDA]
    Justificativa: [Justificativa detalhada da nota, explicando como cada aspecto da análise se relaciona com os critérios de pontuação]
    """
    resposta_nota = generate_rag_response(prompt_nota, [], "competency2")
    return extrair_nota_e_justificativa(resposta_nota)

def atribuir_nota_competency3(analise: str, erros: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_nota = f"""
    Com base na seguinte análise da Competência 3 (Seleção e Organização das Informações) do ENEM, atribua uma nota de 0 a 200 em intervalos de 40 pontos (0, 40, 80, 120, 160 ou 200).

    Análise:
    {analise}

    Considere cuidadosamente os seguintes critérios para atribuir a nota:

    Nota 200:
    - Ideias progressivas e argumentos bem selecionados, revelando um planejamento claro do texto.
    - Apresenta informações, fatos e opiniões relacionados ao tema proposto e aos seus argumentos, de forma consistente e organizada, em defesa de um ponto de vista.
    - Demonstra autoria, com informações e argumentos originais que reforçam o ponto de vista do aluno.
    - Mantém o encadeamento das ideias, com cada parágrafo apresentando informações coerentes com o anterior, sem repetições desnecessárias ou saltos temáticos.
    - Apresenta poucas falhas, e essas falhas não prejudicam a progressão do texto.

    Nota 160:
    - Apresenta informações, fatos e opiniões relacionados ao tema, de forma organizada, com indícios de autoria em defesa de um ponto de vista.
    - Ideias claramente organizadas, mas não tão consistentes quanto o esperado para uma argumentação mais sólida.
    - Organização geral das ideias é boa, mas algumas informações e opiniões não estão bem desenvolvidas.

    Nota 120:
    - Apresenta informações, fatos e opiniões relacionados ao tema, mas limitados aos argumentos dos textos motivadores e pouco organizados, em defesa de um ponto de vista.
    - Ideias previsíveis, sem desenvolvimento profundo ou originalidade, com pouca evidência de autoria.
    - Argumentos simples, sem clara progressão de ideias, e baseado principalmente nas sugestões dos textos motivadores.

    Nota 80:
    - Apresenta informações, fatos e opiniões relacionados ao tema, mas de forma desorganizada ou contraditória, e limitados aos argumentos dos textos motivadores.
    - Ideias não estão bem conectadas, demonstrando falta de coerência e organização no desenvolvimento do texto.
    - Argumentos inconsistentes ou contraditórios, prejudicando a defesa do ponto de vista.
    - Perde linhas com informações irrelevantes, repetidas ou excessivas.

    Nota 40:
    - Apresenta informações, fatos e opiniões pouco relacionados ao tema, com incoerências, e sem defesa clara de um ponto de vista.
    - Falta de organização e ideias dispersas, sem desenvolvimento coerente.
    - Não apresenta um ponto de vista claro, e os argumentos são fracos ou desconexos.

    Nota 0:
    - Apresenta informações, fatos e opiniões não relacionados ao tema, sem coerência, e sem defesa de um ponto de vista.
    - Ideias totalmente desconexas, sem organização ou relação com o tema proposto.
    - Não desenvolve qualquer argumento relevante ou coerente, demonstrando falta de planejamento.

    Forneça a nota e uma justificativa detalhada, relacionando diretamente com a análise fornecida. Certifique-se de que a justificativa esteja completamente alinhada com a nota atribuída e os critérios específicos.

    Formato da resposta:
    Nota: [NOTA ATRIBUÍDA]
    Justificativa: [Justificativa detalhada da nota, explicando como cada aspecto da análise se relaciona com os critérios de pontuação]
    """
    resposta_nota = generate_rag_response(prompt_nota, [], "competency3")
    return extrair_nota_e_justificativa(resposta_nota)

def atribuir_nota_competency4(analise: str, erros: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_nota = f"""
    Com base na seguinte análise da Competência 4 (Conhecimento dos Mecanismos Linguísticos) do ENEM, atribua uma nota de 0 a 200 em intervalos de 40 pontos (0, 40, 80, 120, 160 ou 200).

    Análise:
    {analise}

    Considere cuidadosamente os seguintes critérios para atribuir a nota:

    Nota 200:
    - Utiliza conectivos em todo início de período.
    - Articula bem as partes do texto e apresenta um repertório diversificado de recursos coesivos, conectando parágrafos e períodos de forma fluida.
    - Utiliza referenciação adequada, com pronomes, sinônimos e advérbios, garantindo coesão e clareza.
    - Apresenta transições claras e bem estruturadas entre as ideias de causa/consequência, comparação e conclusão, sem falhas.
    - Demonstra excelente organização de períodos complexos, com uma articulação eficiente entre orações.
    - Não repete muitos conectivos ao longo do texto.

    Nota 160:
    - Deixa de usar uma ou duas vezes conectivos ao longo do texto.
    - Articula as partes do texto, mas com poucas inadequações ou problemas pontuais na conexão de ideias.
    - Apresenta um repertório diversificado de recursos coesivos, mas com algumas falhas no uso de pronomes, advérbios ou sinônimos.
    - As transições entre parágrafos e ideias são adequadas, mas com pequenos deslizes na estruturação dos períodos complexos.
    - Mantém boa coesão e coerência, mas com algumas falhas na articulação entre causas, consequências e exemplos.

    Nota 120:
    - Não usa muitos conectivos ao longo dos parágrafos.
    - Repete várias vezes o mesmo conectivo ao longo do parágrafo.
    - Articula as partes do texto de forma mediana, apresentando inadequações frequentes na conexão de ideias.
    - O repertório de recursos coesivos é pouco diversificado, com uso repetitivo de pronomes.
    - Apresenta transições previsíveis e pouco elaboradas, prejudicando o encadeamento lógico das ideias.
    - A organização dos períodos é mediana, com algumas orações mal articuladas, comprometendo a fluidez do texto.

    Nota 80:
    - Articula as partes do texto de forma insuficiente, com muitas inadequações no uso de conectivos e outros recursos coesivos.
    - O repertório de recursos coesivos é limitado, resultando em repetição excessiva ou uso inadequado de pronomes e advérbios.
    - Apresenta conexões falhas entre os parágrafos, com transições abruptas e pouco claras entre as ideias.
    - Os períodos complexos estão mal estruturados, com orações desconectadas ou confusas.

    Nota 40:
    - Articula as partes do texto de forma precária, com sérias falhas na conexão de ideias.
    - O repertório de recursos coesivos é praticamente inexistente, sem o uso adequado de pronomes, conectivos ou advérbios.
    - Apresenta parágrafos desarticulados, sem relação clara entre as ideias.
    - Os períodos são curtos e desconectados, sem estruturação adequada ou progressão de ideias.

    Nota 0:
    - Não articula as informações e as ideias parecem desconexas e sem coesão.
    - O texto não apresenta recursos coesivos, resultando em total falta de conexão entre as partes.
    - Os parágrafos e períodos são desorganizados, sem qualquer lógica na apresentação das ideias.
    - O texto não utiliza mecanismos de coesão (pronomes, conectivos, advérbios), tornando-o incompreensível.

    Forneça a nota e uma justificativa detalhada, relacionando diretamente com a análise fornecida. Certifique-se de que a justificativa esteja completamente alinhada com a nota atribuída e os critérios específicos.

    Formato da resposta:
    Nota: [NOTA ATRIBUÍDA]
    Justificativa: [Justificativa detalhada da nota, explicando como cada aspecto da análise se relaciona com os critérios de pontuação]
    """
    resposta_nota = generate_rag_response(prompt_nota, [], "competency4")
    return extrair_nota_e_justificativa(resposta_nota)

def atribuir_nota_competency5(analise: str, erros: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_nota = f"""
    Com base na seguinte análise detalhada da Competência 5 (Proposta de Intervenção) do ENEM, atribua uma nota de 0 a 200 em intervalos de 40 pontos (0, 40, 80, 120, 160 ou 200).

    Análise detalhada:
    {analise}

    Considere os seguintes critérios para atribuir a nota:

    Nota 200:
    - Elabora proposta de intervenção completa com todos os 5 elementos (agente, ação, modo/meio, detalhamento e finalidade).

    Nota 160:
    - Elabora bem a proposta de intervenção, mas com apenas 4 elementos presentes.

    Nota 120:
    - Elabora uma proposta de intervenção mediana, com apenas 3 elementos presentes.

    Nota 80:
    - Elabora uma proposta de intervenção insuficiente, com apenas 2 elementos presentes, ou se a proposta for mal articulada ao tema.

    Nota 40:
    - Apresenta uma proposta de intervenção vaga ou precária, apresentando apenas 1 de todos os elementos exigidos.

    Nota 0:
    - Não apresenta proposta de intervenção ou a proposta é completamente desconectada do tema.

    Formato da resposta:
    Nota: [NOTA ATRIBUÍDA]
    Justificativa: [Breve justificativa da nota baseada na análise]
    """
    resposta_nota = generate_rag_response(prompt_nota, [], "competency5")
    return extrair_nota_e_justificativa(resposta_nota)

def extrair_nota_e_justificativa(resposta: str) -> Dict[str, Any]:
   """
   Extrai a nota e justificativa da resposta gerada.
   
   Args:
       resposta: String contendo a resposta completa
       
   Returns:
       Dict contendo nota (int) e justificativa (str)
   """
   linhas = resposta.strip().split('\n')
   nota = None
   justificativa = []
   
   lendo_justificativa = False
   
   for linha in linhas:
       linha = linha.strip()
       if linha.startswith('Nota:'):
           try:
               nota = int(linha.split(':')[1].strip())
           except ValueError:
               raise ValueError("Formato de nota inválido")
       elif linha.startswith('Justificativa:'):
           lendo_justificativa = True
       elif lendo_justificativa and linha:
           justificativa.append(linha)
   
   if nota is None:
       raise ValueError("Nota não encontrada na resposta")
       
   return {
       'nota': nota,
       'justificativa': ' '.join(justificativa)
   }

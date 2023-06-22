Link para o notebook no colab: <a href="https://colab.research.google.com/drive/1FPVrEQy9dMInSYPAoLhp29sQc9lWQsVN?usp=sharing">Notebook</a>
## Análise de Dados e Criação de um Modelo de Machine Learning para Previsão de Depressão com Base no Estudo sobre Hábitos Saudáveis

## Introdução

A depressão é um transtorno que representa um sério problema de saúde pública, sendo uma das principais causas de doenças e debilitações, de acordo com a Organização Mundial da Saúde (OMS). Embora o acesso a diagnósticos e tratamentos esteja aumentando, muitas pessoas ainda não possuem controle adequado dos sintomas depressivos. Para promover um estilo de vida saudável, a OMS recomenda a prática regular de exercícios físicos e a adoção de uma dieta saudável.

Neste artigo, utilizaremos dados de uma pesquisa realizada nos Estados Unidos para analisar o perfil de indivíduos com sintomas depressivos e investigar a associação entre hábitos saudáveis de alimentação, atividade física e a ocorrência de depressão nessa população.

O Estudo Nacional de Exame de Saúde e Nutrição (NHANES)

O NHANES é uma pesquisa conduzida anualmente pelo Centro Nacional de Estatísticas de Saúde (NCHS) dos Centros de Controle e Prevenção de Doenças (CDC) dos Estados Unidos. Essa pesquisa tem como objetivo avaliar a saúde e a nutrição de adultos e crianças no país. Os dados coletados incluem informações demográficas, socioeconômicas, dietéticas, de saúde e também realizam exames médicos, odontológicos, fisiológicos e laboratoriais.

O estudo utiliza uma amostra de aproximadamente 5.000 pessoas a cada ano, selecionadas de forma a representar a população civil não institucionalizada dos EUA. Devido à complexidade da amostragem, é necessário utilizar técnicas e ferramentas adequadas para análise.

## Análise Preliminar de Dados

Inicialmente, foi realizada a junção de dois data frames, denominados df_demo e df_pag, com base na coluna 'SEQN'. O novo data frame resultante, denominado df_full, possui 31 colunas e 5.334 entradas. Em seguida, foi verificado o percentual de valores ausentes em cada coluna, observando-se que a maioria das colunas apresentava cerca de 5,14% de valores nulos. As exceções foram as colunas DPQ010 a DPQ090, com 9,34% de valores ausentes, e a coluna INDFMINC, com 0,88% de valores nulos.

Foram apresentadas medidas resumo das variáveis quantitativas do df_full, excluindo a coluna 'SEQN'. Os valores apresentados incluíam contagem, média, desvio padrão, mínimo, quartis, máximo e percentis (95% e 99%). As variáveis DPQ010 a DPQ090 representam os resultados de um questionário com valores de 0 a 9, em que valores mais altos indicam maior frequência de sintomas depressivos. Já as variáveis HEI2015C1_TOTALVEG a HEI2015_TOTAL_SCORE refletem diferentes aspectos da alimentação saudável, com pontuações variando de 0 a 10.

## Análise de Dados

A análise de dados foi realizada em várias etapas:

Preparação dos dados: Os dados foram preparados por meio de tratamento de variáveis e renomeação de colunas e categorias para melhor compreensão.

Avaliação da frequência das variáveis qualitativas: As frequências das respostas das variáveis qualitativas foram avaliadas por meio da contagem dos valores.

Limpeza e transformações: Foram realizadas transformações nos dados, como substituição de valores ausentes, agrupamento de categorias e substituição de valores categóricos por legendas mais descritivas.

Construção de variáveis: Foram criadas novas variáveis com base nas existentes, como o cálculo do escore PHQ-9 a partir das respostas do questionário, agrupamento das respostas em categorias mais simples e criação de variáveis para identificar a presença de sintomas de depressão.

Análise univariada: Foi realizada uma análise descritiva das variáveis quantitativas e uma análise de frequência das variáveis qualitativas. Também foram criados gráficos para visualizar a distribuição dos dados.

Análise bivariada: Foi realizada uma análise de associação entre a variável de depressão e outras variáveis quantitativas, utilizando gráficos de boxplot. Também foram comparadas as distribuições das variáveis entre os casos positivos de depressão e todos os casos.

## Teste de Hipóteses

Realizamos testes de hipóteses para investigar possíveis diferenças entre grupos de sintomas de depressão em relação a variáveis como idade, renda anual familiar, média de exercícios semanais e escore de saúde. Os resultados revelaram diferenças estatisticamente significativas na renda anual familiar, média de exercícios semanais e escore de saúde entre os grupos de sintomas de depressão.

No caso da idade, não foram encontradas diferenças estatisticamente significativas na média entre os grupos de sintomas de depressão. Esses resultados sugerem que a idade não é um fator determinante para a presença de sintomas depressivos nessa população.

No entanto, observamos diferenças estatisticamente significativas na renda anual familiar entre os grupos de sintomas de depressão. Essa descoberta indica que a renda pode desempenhar um papel importante na predisposição aos sintomas depressivos. É importante considerar que a renda pode afetar fatores como acesso a cuidados de saúde, qualidade de vida e estresse financeiro, que estão relacionados à saúde mental.

Também encontramos diferenças estatisticamente significativas na média de exercícios semanais entre os grupos de sintomas de depressão. Esses resultados sugerem que a prática regular de atividade física pode estar associada a menores índices de depressão. A atividade física tem sido reconhecida como um importante componente na promoção da saúde mental e no alívio dos sintomas depressivos.

Além disso, identificamos diferenças estatisticamente significativas nos escores de saúde entre os grupos de sintomas de depressão. Isso indica que a percepção subjetiva da saúde está relacionada à presença de sintomas depressivos. Essa descoberta destaca a importância de abordar a saúde de forma holística, considerando não apenas aspectos físicos, mas também aspectos emocionais e mentais.

## Análise de Interações entre as Variáveis

Realizamos análises de correlação de ponto bisserial para avaliar as associações entre as variáveis contínuas e a variável alvo "Depressão". Identificamos as variáveis que apresentaram maior associação com a depressão em um contexto de variável alvo binária.

Os resultados revelaram que a renda anual familiar, a escolaridade (especificamente o nível de educação superior completo ou maior), o gênero, a média de atividades físicas semanais e o escore de saúde apresentaram associação significativa com a depressão. Valores negativos indicaram uma associação negativa, enquanto valores positivos indicaram uma associação positiva. Essas variáveis podem ser consideradas relevantes para a previsão da depressão.

Essas análises de interações entre as variáveis forneceram insights importantes sobre os fatores que estão associados à depressão. A renda, a educação, o gênero, a atividade física e a percepção da saúde são aspectos multifacetados que podem influenciar a manifestação e o desenvolvimento da depressão. Compreender essas interações é fundamental para uma abordagem mais abrangente e eficaz no tratamento e prevenção da depressão.

## Modelagem

Com base nos resultados obtidos, desenvolvemos um modelo de Machine Learning utilizando o algoritmo Extra Trees Classifier para prever a depressão. O desempenho do modelo foi avaliado por meio de diversas métricas.

O modelo apresentou uma alta acurácia de 94,98%, o que indica que o modelo está acertando aproximadamente 94,98% das previsões. Além disso, o AUC (Area Under the Curve) de 0,9571 sugere que o modelo possui uma boa capacidade de distinguir entre as classes positiva e negativa.

As métricas de recall, precisão e F1-score revelaram que o modelo é capaz de identificar corretamente cerca de 71,76% dos casos de depressão, enquanto a taxa de casos identificados corretamente como positivos é de aproximadamente 92,46%. O F1-score de 0,8080 indica um bom equilíbrio entre precisão e recall.

O coeficiente Kappa de 0,7796 indica uma concordância considerável além da concordância esperada ao acaso, enquanto o coeficiente MCC (Matthews Correlation Coefficient) de 0,7879 revela um bom desempenho geral do modelo.

## Conclusão

Com base nos resultados e nas saídas dos testes, concluímos que o modelo Extra Trees Classifier apresentou um desempenho promissor na detecção de depressão. Suas métricas de desempenho, como alta acurácia, capacidade de distinguir entre as classes positiva e negativa e equilíbrio entre precisão e recall, indicam seu potencial como uma ferramenta útil para auxiliar na identificação de indivíduos com sintomas depressivos.

No entanto, é importante ressaltar que este modelo foi desenvolvido com base nos dados específicos do estudo e requer validação adicional em outras populações e contextos clínicos. Além disso, é necessário considerar outras variáveis e fatores relevantes para uma compreensão mais abrangente da depressão e sua prevenção.

Em suma, este estudo demonstrou a importância de análises de dados e modelos de Machine Learning na investigação dos fatores associados à depressão e na criação de ferramentas preditivas. Essas abordagens podem fornecer insights valiosos para profissionais de saúde e pesquisadores, contribuindo para a identificação precoce, tratamento adequado e prevenção da depressão, visando promover uma melhor qualidade de vida e bem-estar mental para indivíduos afetados por essa condição.


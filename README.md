# Diagnóstico de Cancro da Mama com Active Learning e Human-in-the-Loop

Projeto da unidade curricular **Inteligência Artificial Centrada no Humano**, Mestrado em Engenharia e Ciência de Dados, Faculdade de Ciências e Tecnologia da Universidade de Coimbra.

**Autores:** Eduardo Cardoso (2021226956), Mafalda Nunes (2020236933), Matilde Reis (2021237887)

## Objetivo

Identificar casos de cancro da mama de forma eficaz a partir de ecografias mamárias, utilizando uma abordagem baseada em **Active Learning** com integração de **Human-in-the-Loop (HitL)**.

Consideramos um trabalho bem-sucedido aquele que resulta na diminuição do número de falsos negativos — ou seja, garantir que o modelo de IA consiga identificar corretamente os casos de cancro, minimizando os erros de não deteção.

Ao aplicar Active Learning, inserimos o ser humano no sistema para que este possa monitorizar o modelo com base no conhecimento do mundo real, ajustando o seu comportamento. A validação humana permite a revisão e validação das decisões críticas feitas pelo modelo, garantindo maior precisão e confiabilidade, além de assegurar supervisão ética e alinhamento dos resultados com os valores humanos. Esta abordagem reforça a confiança na IA ao incorporar o conhecimento e julgamento humano no processo de decisão, e reduz o risco de a IA agir de forma independente em situações que possam entrar em conflito com as expectativas humanas.

Esperamos maximizar a eficiência do modelo, tornando o diagnóstico mais preciso, e acreditamos que esta metodologia pode também contribuir para reduzir a mortalidade ao melhorar a deteção precoce e a tomada de decisão médica, combinando a análise da IA com a decisão humana.

## Estado atual

- Pré-processamento e balanceamento do dataset de ecografias mamárias (classes: `benign`, `malignant`, `normal`).
- CNN para classificação das imagens.
- Active Learning com [modAL](https://modal-python.readthedocs.io/), comparando estratégias de query: `uncertainty_sampling`, `entropy_sampling` e `margin_sampling` (melhor resultado até agora), em diferentes tamanhos de batch.
- Notebook de baseline com Passive Learning para comparação.

## Próximos passos

- **Explicabilidade (XAI)**: introduzir técnicas como Grad-CAM / SHAP para que o humano no loop compreenda *porque* o modelo classificou uma imagem como benigna/maligna, e não apenas a sua confiança.
- Reforçar o componente Human-in-the-Loop (interface/fluxo de validação humana das amostras selecionadas pelo Active Learning).
- Métricas focadas na redução de falsos negativos (não só accuracy).

> Esta secção reflete a direção atual do projeto e será atualizada à medida que for implementada.

## Estrutura do repositório

```
.
├── docs/                     # Documentos do projeto
│   ├── Enunciado.pdf         # Enunciado da UC
│   ├── Proposta_Projeto.pdf  # Proposta entregue
│   ├── Paper.pdf             # Artigo do projeto
│   ├── Resultados.pdf        # Resultados obtidos
│   ├── Apresentação.pdf      # Apresentação do projeto
│   └── Exemplos/             # Artigos de referência (HitL / Active Learning em saúde)
├── notebooks/
│   ├── Code.ipynb                  # Pipeline principal: CNN + Active Learning
│   └── Code_PassiveLearning.ipynb  # Baseline de Passive Learning
└── data/                      # Dataset (não versionado, ver secção abaixo)
```

## Dataset

Usamos o dataset [Breast Ultrasound Images](https://datasetninja.com/breast-ultrasound-images) (licença [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)), com ecografias mamárias anotadas para segmentação e classificação (`benign`, `malignant`, `normal`).

O dataset **não está incluído no repositório** (é demasiado pesado para versionar em git). Para reproduzir os notebooks:

1. Descarregar o dataset a partir do link acima.
2. Colocar o conteúdo em `data/breast-ultrasound-images-DatasetNinja/`, mantendo a estrutura `ds/img/` e `ds/ann/` fornecida pelo DatasetNinja.

## Como correr

Os notebooks foram desenvolvidos para o Google Colab (importam `google.colab.drive`). Para correr localmente:

1. Colocar o dataset em `data/` conforme descrito acima e ajustar os caminhos de leitura das imagens no notebook.
2. Instalar as dependências: `tensorflow`, `modAL-python`, `numpy`, `scikit-learn`, `matplotlib` (e outras usadas ao longo do notebook).
3. Abrir `notebooks/Code.ipynb` num Jupyter/Colab e correr as células sequencialmente.

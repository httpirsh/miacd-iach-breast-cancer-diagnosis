# Diagnóstico de Cancro da Mama com Active Learning e Human-in-the-Loop

Projeto da unidade curricular **Inteligência Artificial Centrada no Humano**, Mestrado em Engenharia e Ciência de Dados, Faculdade de Ciências e Tecnologia da Universidade de Coimbra.

## Objetivo

Identificar casos de cancro da mama a partir de ecografias mamárias, com foco em reduzir falsos negativos. Usamos **Active Learning** com **Human-in-the-Loop (HitL)**: o humano valida as decisões críticas do modelo, aumentando a confiabilidade e garantindo supervisão ética do sistema.

Projeto em fase inicial — a abordagem e o conteúdo deste README vão evoluir.

## Estado atual

- Pré-processamento e balanceamento do dataset (classes: `benign`, `malignant`, `normal`).
- CNN para classificação das imagens.
- Active Learning com [modAL](https://modal-python.readthedocs.io/): comparação das estratégias `uncertainty_sampling`, `entropy_sampling` e `margin_sampling`.
- Baseline com Passive Learning para comparação.

## Próximos passos

- Explicabilidade com Grad-CAM.
- Agente para o paciente: protótipo num notebook que explica a previsão e responde a perguntas do domínio com RAG, usando uma API de LLM.

## Estrutura do repositório

```
.
├── notebooks/                       # correr por esta ordem
│   ├── 00_data_exploration.ipynb    # classes, exemplos, tamanhos, memória necessária
│   ├── 01_passive_learning.ipynb    # baseline: CNN treinada com todo o conjunto de treino
│   └── 02_active_learning.ipynb     # Active Learning (uncertainty, entropy, margin) vs baseline
├── src/                             # código partilhado pelos notebooks
│   ├── config.py                    # caminhos (dataset, resultados)
│   ├── data.py                      # carregamento, divisão treino/teste e balanceamento
│   ├── exploration.py               # gráficos e contagens da exploração dos dados
│   ├── model.py                     # CNN
│   ├── active_learning.py           # ciclo de Active Learning com Human-in-the-Loop
│   └── evaluation.py                # métricas (recall, F1, falsos negativos) e gráficos
├── docs/
│   ├── projeto/                     # enunciado 2026/27, a nossa proposta, checklist, análise
│   ├── referencia_grupo_2024/       # trabalho de outro grupo sobre o mesmo tema (orientação)
│   ├── exemplos_artigos/            # artigos de exemplo indicados pelo professor
│   ├── template_ijcai26/            # template do relatório (IJCAI-ECAI 26)
│   └── dataset/                     # metadados e licença do dataset
├── results/                         # métricas guardadas pelos notebooks (criada ao correr)
└── data/                            # dataset (não versionado, ver abaixo)
```

## Dataset

[Breast Ultrasound Images](https://datasetninja.com/breast-ultrasound-images) (CC0 1.0). Não está incluído no repositório por ser demasiado pesado. Para reproduzir os notebooks, descarregar e colocar em `data/breast-ultrasound-images-DatasetNinja/`, mantendo a estrutura `ds/img/` e `ds/ann/`.

## Como correr

Correr os notebooks por ordem: `00` (exploração, opcional), `01` (guarda a baseline em `results/`) e `02`. Os parâmetros (tamanho das imagens, nº de imagens por classe, batch sizes, nº de iterações, humano real ou simulado) estão na secção **1. Configuração** de cada notebook.

- **Google Colab:** copiar o repositório (com o dataset em `data/`) para o Google Drive, em `My Drive/miacd-iach-breast-cancer-diagnosis`. O notebook monta o Drive e instala as dependências em falta.
- **Localmente:** uma versão de Python suportada pelo TensorFlow (ex.: 3.11; o 3.14 ainda não é suportado) e `pip install -r requirements.txt`. Abrir os notebooks a partir da pasta `notebooks/`.

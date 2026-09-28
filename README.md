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

- Explicabilidade (ex.: Grad-CAM, SHAP).
- Componente Human-in-the-Loop (validação humana das amostras selecionadas).

## Estrutura do repositório

```
.
├── docs/          # Enunciado, proposta, paper, resultados, apresentação, referências
├── notebooks/      # Code.ipynb (Active Learning) e Code_PassiveLearning.ipynb (baseline)
└── data/           # Dataset (não versionado, ver abaixo)
```

## Dataset

[Breast Ultrasound Images](https://datasetninja.com/breast-ultrasound-images) (CC0 1.0). Não está incluído no repositório por ser demasiado pesado. Para reproduzir os notebooks, descarregar e colocar em `data/breast-ultrasound-images-DatasetNinja/`, mantendo a estrutura `ds/img/` e `ds/ann/`.

## Como correr

Notebooks desenvolvidos para o Google Colab. Localmente: colocar o dataset em `data/`, ajustar os caminhos no notebook e instalar `tensorflow`, `modAL-python`, `numpy`, `scikit-learn`, `matplotlib`.

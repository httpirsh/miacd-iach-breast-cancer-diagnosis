# Diagnóstico de Cancro da Mama com Active Learning e Human-in-the-Loop

Projeto da unidade curricular **Inteligência Artificial Centrada no Humano**, Mestrado em Engenharia e Ciência de Dados, Faculdade de Ciências e Tecnologia da Universidade de Coimbra.

## Objetivo

Identificar casos de cancro da mama a partir de ecografias mamárias, com foco em reduzir falsos negativos. Usamos **Active Learning** com **Human-in-the-Loop (HitL)**: o humano valida as decisões críticas do modelo, aumentando a confiabilidade e garantindo supervisão ética do sistema.

Projeto em fase inicial — a abordagem e o conteúdo deste README vão evoluir.

## Estado atual

- Pré-processamento sem data leakage: o dataset tem 780 ficheiros mas só **601 ecografias diferentes** (muitas estão repetidas com recortes ligeiramente diferentes). A divisão treino/validação/teste (~65/15/20) é feita **por grupos de imagens repetidas** e **antes** do balanceamento; 19 imagens repetidas com labels contraditórias são excluídas.
- **Passive Learning** (notebook 01, média de 3 seeds no teste):

  | Modelo | Accuracy | F1 macro | Recall maligno |
  |---|---|---|---|
  | CNN treinada de raiz (lr 0.001 + early stopping) | 0.63 | 0.61 | 0.68 |
  | MobileNetV2 pré-treinada + fine-tuning | **0.79** | **0.76** | **0.69** |

- **Active Learning** com [modAL](https://modal-python.readthedocs.io/) (notebook 02): `uncertainty`, `entropy`, `margin` e uma estratégia nossa focada nos malignos, comparadas com a escolha ao acaso (20 seeds). Sem data leakage, **nenhuma estratégia poupa anotações em relação à escolha ao acaso**, e as estratégias de incerteza detetam um pouco menos cancros no início.
- Exemplos de previsões do modelo (falsos negativos, outros erros, acertos) no notebook 01.
- Human-in-the-Loop: o humano pode dar as labels no notebook 02 (`HUMAN = True`).

## Próximos passos

- Explicabilidade com Grad-CAM.
- Agente para o paciente: protótipo num notebook que explica a previsão e responde a perguntas do domínio com RAG, usando uma API de LLM.

## Estrutura do repositório

```
.
├── notebooks/                       # correr por esta ordem
│   ├── 00_data_exploration.ipynb    # classes, exemplos, tamanhos, memória necessária
│   ├── 01_passive_learning.ipynb    # baseline: CNN e MobileNetV2 treinadas com todo o conjunto de treino
│   └── 02_active_learning.ipynb     # Active Learning (uncertainty, entropy, margin) vs escolha ao acaso
├── src/                             # código partilhado pelos notebooks
│   ├── config.py                    # caminhos (dataset, resultados)
│   ├── data.py                      # carregamento, divisão treino/teste e balanceamento
│   ├── exploration.py               # gráficos e contagens da exploração dos dados
│   ├── gpu.py                       # preparar a GPU NVIDIA local para o TensorFlow
│   ├── model.py                     # CNN, MobileNetV2 (transfer learning), treino com early stopping
│   ├── active_learning.py           # ciclo de Active Learning com Human-in-the-Loop
│   └── evaluation.py                # métricas (recall, F1, falsos negativos) e gráficos
├── docs/
│   ├── projeto/                     # enunciado 2026/27, a nossa proposta, checklist, análise
│   ├── exemplos_artigos/            # artigos de exemplo indicados pelo professor
│   ├── template_ijcai26/            # template do relatório (IJCAI-ECAI 26)
│   └── dataset/                     # metadados e licença do dataset
├── report/                          # relatório final em LaTeX (template IJCAI-ECAI 26)
├── results/                         # métricas guardadas pelos notebooks (criada ao correr)
└── data/                            # dataset (não versionado, ver abaixo)
```

## Dataset

[Breast Ultrasound Images](https://datasetninja.com/breast-ultrasound-images) (CC0 1.0). Não está incluído no repositório por ser demasiado pesado. Para reproduzir os notebooks, descarregar e colocar em `data/breast-ultrasound-images-DatasetNinja/`, mantendo a estrutura `ds/img/` e `ds/ann/`.

## Como correr

Correr os notebooks por ordem: `00` (exploração, opcional), `01` (guarda a baseline em `results/`) e `02`. Os parâmetros (tamanho das imagens, learning rate, seeds, nº de imagens pedidas ao humano, humano real ou simulado) estão na secção **1. Configuração** de cada notebook.

### Localmente, com GPU

É preciso uma versão de Python suportada pelo TensorFlow (ex.: 3.11; o 3.14 ainda não é suportado) e, para usar a GPU, uma placa NVIDIA com o driver instalado. No Linux, o `requirements.txt` instala o TensorFlow com as bibliotecas CUDA incluídas, sem ser preciso instalar o CUDA à parte.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# confirmar que a GPU é detetada (deve dizer "A treinar na GPU: ...")
python -c "from src.gpu import setup_gpu; setup_gpu()"
```

Registar o kernel do Jupyter (uma vez) e abrir os notebooks a partir da pasta `notebooks/`, com o kernel **IACH (.venv)**:

```bash
python -m ipykernel install --user --name iach-projeto --display-name "IACH (.venv)"
```

A primeira célula de cada notebook de treino chama `setup_gpu()` (`src/gpu.py`), que prepara a GPU e diz se o treino vai correr na GPU ou no CPU. Se a GPU ficar sem memória, reduzir o `IMG_SIZE`. Na primeira execução, o Keras descarrega os pesos pré-treinados da MobileNetV2 (~9 MB).

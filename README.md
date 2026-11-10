# Diagnóstico de Cancro da Mama com Active Learning e Human-in-the-Loop

Projeto da unidade curricular **Inteligência Artificial Centrada no Humano**, Mestrado em Engenharia e Ciência de Dados, Faculdade de Ciências e Tecnologia da Universidade de Coimbra.

## Objetivo

Identificar casos de cancro da mama a partir de ecografias mamárias, com foco em reduzir falsos negativos. Usamos **Active Learning** com **Human-in-the-Loop (HitL)**: o humano valida as decisões críticas do modelo, aumentando a confiabilidade e garantindo supervisão ética do sistema.

## Estado atual

- Pré-processamento sem data leakage: o dataset tem 780 ficheiros mas só **601 ecografias diferentes** (muitas estão repetidas com recortes ligeiramente diferentes). A divisão treino/validação/teste (~64/17/20) é feita **por grupos de imagens repetidas** e **antes** do balanceamento; 19 imagens repetidas com labels contraditórias são excluídas.
- **Passive Learning** (notebook 01, 5 seeds, treino reprodutível): a MobileNetV2 pré-treinada chega a ~0.80 de accuracy e a CNN treinada de raiz a ~0.61; ambas detetam ~70% dos cancros. Os números completos estão no notebook 01 e no relatório (`report/`).
- **Active Learning** com [modAL](https://modal-python.readthedocs.io/) (notebook 02): `uncertainty`, `margin` e uma estratégia nossa focada nos malignos, comparadas com a escolha ao acaso (20 seeds). Sem data leakage, **nenhuma estratégia poupa anotações em relação à escolha ao acaso**, e as estratégias de incerteza detetam um pouco menos cancros no início.
- Exemplos de previsões do modelo (falsos negativos, outros erros, acertos) no notebook 01.
- **Explicabilidade com Grad-CAM** (notebook 03): mapas de onde a MobileNetV2 olhou, comparados com o contorno das lesões desenhado pelos médicos. Quando deteta um cancro, o modelo olha sobretudo para a lesão; nos falsos negativos, umas vezes olha para a lesão e acha-a benigna, outras olha para o tecido ao lado. Enviar ao médico os casos com confiança abaixo de 90% apanharia 7 dos 10 falsos negativos, à custa de rever 41% das imagens.
- Human-in-the-Loop: o humano pode dar as labels no notebook 02 (`HUMAN = True`).
- **Protótipo do agente** (notebook 04): classifica a ecografia, mostra o Grad-CAM e decide para onde vai o caso. Se o modelo prevê maligno ou tem confiança abaixo de um limiar escolhido na validação, o caso vai sempre para o médico (Human-in-the-Loop); os outros ficam num log que o médico revê e corrige (Human-on-the-Loop). Tem um chatbot com RAG sobre documentos médicos, ligado ao LLM do [IAedu](https://iaedu.pt).
- **Documentos do chatbot** (`docs/rag/`): 19 textos para pacientes, em português, do SNS 24, da DGS, da Liga Portuguesa Contra o Cancro, de hospitais do SNS (São João, Santa Maria) e da Sociedade Portuguesa de Radiologia. Cobrem rastreio, ecografia e mamografia, BI-RADS, benigno e maligno, biópsia, limites dos exames e tratamento. A lista com os endereços e o que foi retirado de cada um está em `docs/rag/fontes.md`.

## Próximos passos

- Textos finais para o paciente e avaliação das respostas do chatbot (ver as issues).

## Estrutura do repositório

```
.
├── notebooks/                       # correr por esta ordem
│   ├── 00_data_exploration.ipynb    # classes, exemplos, tamanhos, memória necessária
│   ├── 01_passive_learning.ipynb    # baseline: CNN e MobileNetV2 treinadas com todo o conjunto de treino
│   ├── 02_active_learning.ipynb     # Active Learning (uncertainty, margin, malignant) vs escolha ao acaso
│   ├── 03_explicabilidade.ipynb     # Grad-CAM: onde o modelo olhou, comparado com as lesões
│   └── 04_agente.ipynb              # protótipo do agente: decisão, médico no ciclo, log e chatbot
├── src/                             # código partilhado pelos notebooks
│   ├── config.py                    # caminhos (dataset, resultados, modelos)
│   ├── data.py                      # carregamento, divisão treino/teste e balanceamento
│   ├── exploration.py               # gráficos e contagens da exploração dos dados
│   ├── gpu.py                       # preparar a GPU NVIDIA local para o TensorFlow
│   ├── model.py                     # CNN, MobileNetV2 (transfer learning), treino com early stopping
│   ├── active_learning.py           # ciclo de Active Learning com Human-in-the-Loop
│   ├── explain.py                   # Grad-CAM e comparação com as máscaras das lesões
│   ├── agent.py                     # regra para chamar o médico, log de casos e textos para o paciente
│   ├── rag.py                       # RAG: divisão dos documentos e pesquisa (TF-IDF)
│   ├── llm.py                       # ligação ao LLM do IAedu
│   └── evaluation.py                # métricas (accuracy, recall de maligno, falsos negativos) e gráficos
├── docs/
│   ├── projeto/                     # enunciado 2026/27, a nossa proposta, checklist, análise
│   ├── rag/                         # documentos médicos usados pelo chatbot (fontes em fontes.md)
│   └── dataset/                     # metadados e licença do dataset
├── report/                          # relatório final em LaTeX (template IJCAI-ECAI 26)
├── results/                         # métricas guardadas pelos notebooks (criada ao correr)
├── models/                          # MobileNetV2 treinada pelo notebook 01 (não versionada)
└── data/                            # dataset (não versionado, ver abaixo)
```

## Dataset

[Breast Ultrasound Images](https://datasetninja.com/breast-ultrasound-images) (CC0 1.0). Não está incluído no repositório por ser demasiado pesado. Para reproduzir os notebooks, descarregar e colocar em `data/breast-ultrasound-images-DatasetNinja/`, mantendo a estrutura `ds/img/` e `ds/ann/`.

## Como correr

Correr os notebooks por ordem: `00` (exploração, opcional), `01` (guarda a baseline em `results/` e o modelo em `models/`), `02`, `03` e `04` (usam o modelo guardado pelo `01`). Os parâmetros (tamanho das imagens, learning rate, seeds, nº de imagens pedidas ao humano, humano real ou simulado) estão na secção **1. Configuração** de cada notebook.

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

### Chatbot (notebook 04)

O chatbot usa o LLM do [IAedu](https://iaedu.pt) (acesso com as credenciais da universidade). No IAedu, criar um agente; na conversa modelo do agente, em engrenagem → «Informação da API», copiar a chave, o identificador do agente e o do canal para um ficheiro `.env` na raiz do repositório:

```
IAEDU_API_KEY=...
IAEDU_AGENT_ID=...
IAEDU_CHANNEL_ID=...
```

O `.env` está no `.gitignore`: a chave nunca vai para o git. Sem `.env`, o notebook 04 corre na mesma e o chatbot mostra só a pesquisa nos documentos.

# Guia completo do projeto

**Diagnóstico de cancro da mama em ecografias com Active Learning e Human-in-the-Loop**
Inteligência Artificial Centrada no Humano (IACH / HCAI) 2026/27

Este guia explica o projeto do zero: o problema, cada conceito técnico, o que fizemos, os números,
e as perguntas que podem surgir na apresentação. Os números são os do relatório final
(`report/main.tex`) e dos notebooks.

---

## Índice

1. [O projeto numa página](#1-o-projeto-numa-página)
2. [O problema clínico](#2-o-problema-clínico)
3. [Os dados (BUSI)](#3-os-dados-busi)
4. [O achado das imagens repetidas (data leakage)](#4-o-achado-das-imagens-repetidas-data-leakage)
5. [Divisão dos dados e balanceamento](#5-divisão-dos-dados-e-balanceamento)
6. [Conceitos básicos de deep learning](#6-conceitos-básicos-de-deep-learning)
7. [Passive learning: os dois modelos](#7-passive-learning-os-dois-modelos)
8. [Métricas: porque não chega a accuracy](#8-métricas-porque-não-chega-a-accuracy)
9. [Active Learning](#9-active-learning)
10. [O humano no ciclo (Human-in-the-Loop)](#10-o-humano-no-ciclo-human-in-the-loop)
11. [Explicabilidade: Grad-CAM](#11-explicabilidade-grad-cam)
12. [O agente](#12-o-agente)
13. [O chatbot (RAG + IAedu)](#13-o-chatbot-rag--iaedu)
14. [Ligação à IA Centrada no Humano](#14-ligação-à-ia-centrada-no-humano)
15. [Limitações](#15-limitações)
16. [O código: o que faz cada ficheiro](#16-o-código-o-que-faz-cada-ficheiro)
17. [Os notebooks, passo a passo](#17-os-notebooks-passo-a-passo)
18. [Folha de números](#18-folha-de-números)
19. [Perguntas prováveis e respostas](#19-perguntas-prováveis-e-respostas)
20. [Glossário](#20-glossário)
21. [Sugestão de apresentação](#21-sugestão-de-apresentação)

---

## 1. O projeto numa página

**Pergunta de investigação:** o Active Learning (AL) baseado em incerteza poupa anotações do
médico **sem fazer falhar mais cancros**?

**O que fizemos, por ordem:**

1. Explorámos o dataset BUSI (780 ecografias: normal, benigno, maligno) e descobrimos que tem
   **imagens repetidas**: só há 601 ecografias diferentes.
2. Dividimos os dados **por ecografia** (todas as cópias de uma ecografia no mesmo conjunto), para
   não haver *data leakage*.
3. Treinámos dois modelos com todas as labels (*passive learning*): uma CNN feita de raiz e uma
   MobileNetV2 pré-treinada. A MobileNetV2 é a melhor (accuracy 0.804).
4. Simulámos AL: o modelo começa com 20 imagens anotadas e vai pedindo mais 10 de cada vez.
   Comparámos 4 formas de escolher as imagens (aleatória, least confidence, margin, e uma nossa,
   "malignant"), cada uma repetida 20 vezes.
5. Explicámos as decisões do modelo com **Grad-CAM** e comparámos os mapas com as lesões
   desenhadas pelos médicos.
6. Construímos um **agente** (protótipo num notebook) que decide quando um caso tem de ir ao
   médico, e um **chatbot** que responde a perguntas do paciente com base em documentos médicos.

**Resultados principais:**

- As imagens repetidas inflacionavam tudo: com a divisão errada (por ficheiro), a MobileNetV2 tinha
  accuracy 0.840 e o AL parecia poupar 24% das anotações. Com a divisão correta, a accuracy desce
  para 0.804 e a poupança **desaparece**.
- Com a divisão correta, **nenhuma estratégia de AL é melhor do que escolher ao acaso**. A least
  confidence até deteta menos cancros no início (58% contra 65% com 100 anotações).
- O Grad-CAM mostra que o modelo olha para a lesão quando deteta um cancro, mas também falha cancros
  **com muita confiança** (3 dos 10 falhados com ≥97%).
- O agente manda ao médico 55% dos casos e 36 dos 39 cancros chegam até ele.

**A história numa frase:** *"O que parecia um ganho era um artefacto dos dados, e o que o modelo
pergunta ao médico importa."*

---

## 2. O problema clínico

- O cancro da mama é o cancro mais diagnosticado nas mulheres e a principal causa de morte por
  cancro entre elas (Bray et al., 2024). Detetá-lo cedo aumenta muito a sobrevivência.
- A **ecografia** é um exame barato, sem radiação e muito disponível. É usada para caracterizar
  lesões e como primeiro exame em mamas densas e mulheres mais novas. Mas interpretá-la depende
  muito da experiência de quem a faz → motiva sistemas de apoio ao diagnóstico.
- **Os erros não custam o mesmo:**
  - **Falso positivo** (dizer maligno a algo benigno): leva a mais exames, ansiedade, talvez uma
    biópsia. Mau, mas recuperável.
  - **Falso negativo** (dizer benigno/normal a um cancro): o cancro pode não ser tratado a tempo.
    **É o erro mais grave.** Por isso olhamos sobretudo para o *recall de maligno* e para os falsos
    negativos.
- **Dois obstáculos** para usar CNNs na clínica:
  1. Precisam de muitas imagens **anotadas**, e cada anotação custa tempo de um especialista.
     → É isto que o **Active Learning** tenta resolver.
  2. Um modelo que decide sozinho é difícil de confiar, auditar e corrigir.
     → Daí o **Human-in-the-Loop**, a **explicabilidade** e o **agente que encaminha para o médico**.

---

## 3. Os dados (BUSI)

- **BUSI** = *Breast Ultrasound Images* (Al-Dhabyani et al., 2020). É o dataset público mais usado
  para este problema. Obtivemo-lo pelo Dataset Ninja (licença CC0, livre).
- **780 imagens PNG** de 600 mulheres entre os 25 e os 75 anos, recolhidas em 2018.
- **3 classes**, dadas pelo nome do ficheiro (ex.: `benign (12).png`):
  - normal: 133
  - benign (benigno): 437
  - malignant (maligno): 210
- **Desequilibrado:** há mais de 3 imagens benignas por cada normal.
- Cada imagem benigna e maligna tem uma **máscara da lesão** desenhada por médicos (usamos isto
  no Grad-CAM para verificar se o modelo olha para a lesão).
- Tamanhos variados: largura 190–1048 px, altura 310–719 px. Por isso redimensionamos todas.
- Algumas imagens têm **anotações do ecógrafo** (texto, marcas, réguas) gravadas na imagem. Um
  modelo pode aprender a olhar para essas marcas em vez da lesão (*shortcut learning*).

---

## 4. O achado das imagens repetidas (data leakage)

**Este é o achado mais forte do projeto.**

### Como foi descoberto
Ao olhar para os **falsos negativos** dos primeiros modelos, vimos imagens de teste visualmente
iguais a imagens de treino. Comparar os ficheiros byte a byte só encontrava **um** duplicado
(`benign (433)` = `malignant (145)`). Os outros eram a mesma ecografia com um **recorte ligeiramente
diferente**, por isso os ficheiros não eram iguais.

> Ponto HCAI: descobrimos o problema porque olhámos para os **erros** do modelo, e não só para as
> métricas.

### Como detetamos as repetidas (`find_duplicates` em `src/data.py`)
1. Cada imagem é reduzida a **32×32 píxeis** (assim pequenas diferenças de recorte quase
   desaparecem).
2. Calcula-se a **correlação** entre os píxeis de cada par de imagens (1 = iguais, 0 = nada a ver).
3. Se a correlação for **> 0.95**, são consideradas a mesma ecografia e ficam no mesmo **grupo**.
4. Se uma imagem é igual a B, e B é igual a C, as três ficam no mesmo grupo.

### O que encontrámos
- **780 ficheiros → só 601 ecografias diferentes.**
  - 462 aparecem 1 vez, 112 aparecem 2 vezes, 20 aparecem 3 vezes, 5 aparecem 4 vezes, 2 aparecem
    7 vezes.
- **19 imagens com labels contraditórias:** a mesma ecografia aparece como benigna num ficheiro e
  maligna noutro (ou benigna e normal). Como não sabemos qual está certa, **excluímo-las**.
- Ficam **761 imagens**: 131 normais, 426 benignas, 204 malignas.

### Porque é que isto é grave: *data leakage*
*Data leakage* (fuga de dados) = o modelo, no treino, tem acesso a informação que não devia ter,
neste caso **cópias das imagens de teste**. O teste deixa de medir "imagens novas" e passa a medir
"imagens que o modelo já viu" → resultados inflacionados.

| | Divisão por ficheiro (errada) | Divisão por ecografia (correta) |
|---|---|---|
| Accuracy MobileNetV2 | 0.840 | 0.804 |
| Recall de maligno | 0.802 | 0.697 |
| Poupança do AL | "24% menos anotações" | nenhuma |

### Porque é que o leakage fazia o AL parecer bom
(Duas explicações plausíveis, que não medimos separadamente.)
1. Se cópias da mesma ecografia estão no treino e no teste, o modelo acerta a cópia do teste "de
   graça". Isto favorece modelos que decoram imagens (a MobileNetV2 tem 1.00 de accuracy no treino).
2. Cerca de 1/5 dos ficheiros (179 de 780) são cópias extra. A escolha **ao acaso** gasta algumas
   anotações em cópias de ecografias já anotadas (não acrescentam informação). As estratégias de
   incerteza evitam-nas, porque o modelo já está confiante numa imagem cuja cópia viu.
   → **O leakage fazia o AL parecer melhor porque fazia a escolha ao acaso parecer pior.**

Na divisão errada, a margin precisava de 190 anotações para se aproximar do resultado com o pool
todo, contra 250 da escolha ao acaso: daí os "24% menos".

---

## 5. Divisão dos dados e balanceamento

### Os três conjuntos
| Conjunto | Para que serve | Tamanho |
|---|---|---|
| **Treino** (~64%) | o modelo aprende com estas imagens | 485 originais |
| **Validação** (~17%) | decidir quando parar o treino, escolher parâmetros (C, limiar τ) | 127 |
| **Teste** (~20%) | medir o resultado final. **Nunca** é usado para decidir nada | 149 (39 malignas) |

**Regra de ouro:** as imagens de validação e de teste **nunca** são usadas no treino (nem cópias
nem versões rodadas delas).

### Como é feita a divisão
- `StratifiedGroupKFold` (scikit-learn), em `load_dataset`:
  - **Stratified** = cada conjunto tem mais ou menos a mesma proporção de classes.
  - **Group** = todas as imagens do mesmo grupo (mesma ecografia) ficam no mesmo conjunto.
  - Com 5 partes: 1 parte (20%) para teste; do resto, 1 de 5 partes (⅕ de 80% ≈ 17%) para
    validação; sobra ~64% para treino.
- A divisão é feita **antes** de qualquer aumento de dados.

| | Normal | Benigno | Maligno | Total |
|---|---|---|---|---|
| Treino (original) | 82 | 275 | 128 | 485 |
| Treino (balanceado) | 492 | 550 | 512 | 1554 |
| Pool do AL (1 por ecografia) | 61 | 202 | 118 | 381 |
| Validação | 21 | 69 | 37 | 127 |
| Teste | 28 | 82 | 39 | 149 |

### Balanceamento (só no treino do passive learning)
Sem balancear, o modelo tende a dizer "benigno" quase sempre (é a classe maioritária). Para
equilibrar, acrescentamos **rotações e reflexões** das imagens do próprio treino (*data
augmentation*):
- normal: +5 cópias de cada (×6)
- maligno: +3 cópias de cada (×4)
- benigno: +1 cópia de cada (×2)

Rodar ou espelhar uma ecografia não muda o diagnóstico, por isso são exemplos válidos.
(Limitação: rotações de 90° e 180° dão imagens que um ecógrafo nunca produz.)

**Validação e teste não são balanceados:** ficam com as imagens originais e a distribuição real.

### No Active Learning é diferente
- **Sem augmentation**: o especialista anota imagens reais, não cópias rodadas.
- **Uma imagem por ecografia** (`unique_train=True`): não faz sentido pedir ao médico para anotar
  a mesma ecografia duas vezes. Por isso o pool tem 381 imagens.

---

## 6. Conceitos básicos de deep learning

- **Rede neuronal:** função com muitos parâmetros (pesos) que transforma uma entrada (a imagem)
  numa saída (as probabilidades das 3 classes). Treinar = ajustar os pesos para errar menos.
- **CNN (rede convolucional):** rede própria para imagens. As camadas **convolucionais** passam
  pequenos filtros (ex.: 4×4) pela imagem e detetam padrões: primeiro bordas e texturas, depois
  formas mais complexas.
- **Max pooling:** reduz o tamanho da imagem guardando o valor máximo de cada zona.
- **ReLU:** função de ativação (`max(0, x)`). Funciona melhor do que a sigmoid em camadas
  escondidas (a sigmoid "satura" e o modelo aprende devagar; foi um dos problemas da nossa
  primeira CNN).
- **Batch normalization:** normaliza os valores entre camadas; torna o treino mais estável.
- **Dropout (0.3):** em cada passo de treino, desliga ao acaso 30% dos neurónios. Evita que o
  modelo decore os dados.
- **Softmax:** última camada; transforma os números de saída em **probabilidades** que somam 1
  (ex.: normal 0.05, benigno 0.80, maligno 0.15). A classe prevista é a de maior probabilidade; essa
  probabilidade é a **confiança** do modelo.
- **Loss (categorical cross-entropy):** mede o quão errado está o modelo; o treino tenta
  minimizá-la.
- **Adam:** o algoritmo que ajusta os pesos. **Learning rate** = o tamanho de cada ajuste
  (0.001). Demasiado pequeno (0.00001) → quase não aprende (aconteceu na nossa primeira CNN).
- **Época:** uma passagem completa por todas as imagens de treino.
- **Mini-batch (32):** os pesos são ajustados a cada grupo de 32 imagens.
- **Overfitting (sobreajuste):** o modelo decora o treino e não generaliza. Sinal: accuracy de
  treino muito maior do que a de validação. Os nossos dois modelos fazem overfitting (treino
  0.96–1.00, validação ≤0.78 / ≤0.86).
- **Early stopping:** para o treino quando a loss de **validação** não melhora durante 10 épocas
  (máximo 60) e volta aos pesos da melhor época. Só a validação decide; o teste não.
- **Seed:** número que fixa o "acaso" (pesos iniciais, ordem das imagens…). Mesma seed → mesmo
  resultado. Usamos 5 seeds no passive learning e 20 no AL, e reportamos a média e o desvio
  padrão, porque um único treino pode ter sorte ou azar.
- **Treino determinístico:** na GPU, algumas operações somam valores por uma ordem que muda de
  corrida para corrida. Ativámos o modo determinístico do TensorFlow (`src/gpu.py`) para a mesma
  seed dar sempre o mesmo modelo → **reprodutibilidade**.

---

## 7. Passive learning: os dois modelos

*Passive learning* = o modelo aprende com **todas** as imagens anotadas, sem escolher nada. É a
referência a bater.

### Pré-processamento
- Escala de cinzento (1 canal; a ecografia é monocromática).
- Redimensionadas para **128×128** e valores em [0, 1].
- Porque 128 e não 224? Com 224 o notebook de AL ficava sem RAM. Em 128×128 as imagens ocupam
  ~120 MB em vez de ~367 MB, e a 224 a MobileNetV2 não era melhor. (Computador: RTX 3050 com 4 GB,
  13 GB de RAM.)

### Modelo 1: CNN treinada de raiz (`build_cnn`)
- 4 blocos convolucionais com 32, 64, 128 e 32 filtros 4×4, cada um com ReLU, batch normalization e
  max pooling; dropout 0.3 depois do 2.º e 3.º blocos; uma camada densa de 32 neurónios; softmax
  com 3 saídas.
- Adam, learning rate 0.001, early stopping.
- **Resultado:** accuracy 0.609 (varia entre 0.55 e 0.71 conforme a seed). Recall de maligno 0.718,
  mas consegue-o porque chama "maligno" a muitas imagens benignas e normais (muitos falsos
  positivos).

### Modelo 2: MobileNetV2 com transfer learning (`build_mobilenet`, `unfreeze_top`)
- **Transfer learning:** em vez de aprender de raiz, usamos uma rede já treinada noutro problema.
  A MobileNetV2 foi treinada no **ImageNet** (>1 milhão de fotografias) e já sabe detetar bordas,
  texturas e formas. Com menos de 400 ecografias diferentes no treino, aprender isso de raiz é
  difícil.
- A MobileNetV2 é pequena (cabe na GPU de 4 GB).
- A rede espera imagens a cores (3 canais) com valores em [-1, 1] → repetimos o canal cinzento 3
  vezes e reescalamos.
- Tiramos a última camada original (1000 classes do ImageNet) e pomos uma nova com as **nossas 3
  classes**.
- **Treino em 2 fases:**
  1. A rede pré-treinada fica **congelada** (não muda); só a nova camada aprende (lr 0.001).
  2. **Fine-tuning:** descongelamos as últimas 30 camadas e continuamos com um learning rate 10×
     menor (0.0001), para não estragar o que a rede já sabe.
- **Resultado:** accuracy **0.804** (o melhor), recall de maligno 0.697, ~11.8 falsos negativos em
  39 (entre 10 e 15 conforme a seed).

### Tabela de resultados (teste, média ± desvio padrão em 5 seeds)
| Modelo | Accuracy | Recall maligno | Falsos negativos |
|---|---|---|---|
| CNN | 0.609 ± 0.060 | 0.718 ± 0.093 | 11.0 ± 3.6 |
| MobileNetV2 | **0.804 ± 0.022** | 0.697 ± 0.050 | 11.8 ± 1.9 |
| MobileNetV2, divisão por ficheiro (com leakage) | 0.840 ± 0.005 | 0.802 ± 0.049 | 8.3 ± 2.1 |

### Matrizes de confusão (somadas nas 5 seeds)
- Uma **matriz de confusão** mostra, para cada classe verdadeira (linhas), o que o modelo previu
  (colunas). Os falsos negativos estão na linha "malignant", colunas "normal" e "benign".
- MobileNetV2: a maioria dos cancros falhados são previstos como **benignos** (50 de 59). Recall
  por classe: normal 0.79, benigno 0.86, maligno 0.70.
- CNN: 40 de 55 cancros falhados vão para benigno e 15 para normal; também confunde muitas imagens
  benignas com normais (80) e malignas (98).

### Experiências preliminares (com uma seed)
- A primeira CNN (lr 0.00001, sigmoid na camada densa) quase não aprendia: accuracy 0.36, pior do
  que dizer sempre "benigno". Corrigido com lr 0.001 e ReLU.
- Na MobileNetV2, o fine-tuning foi melhor do que a rede congelada.
- 224×224 não foi melhor do que 128×128.
- Dar mais peso aos erros em malignos na loss não aumentou o recall de maligno (provavelmente
  porque o treino já está balanceado).
- Limitação: estas escolhas foram feitas olhando para o teste da primeira divisão.

---

## 8. Métricas: porque não chega a accuracy

Usamos 3 métricas no **teste** (`compute_metrics` em `src/evaluation.py`):

- **Accuracy** = % de imagens bem classificadas. Problema: num dataset desequilibrado, pode ser alta
  mesmo falhando muitos cancros.
- **Recall de maligno** (sensibilidade) = dos cancros que existem, quantos o modelo deteta.
  `recall = cancros detetados / cancros totais`. Ex.: 0.70 → deteta 70% dos cancros, falha 30%.
- **Falsos negativos (FN)** = número de cancros classificados como benigno ou normal.
  Com 39 malignos no teste, **1 falso negativo muda o recall em 2.6 pontos** (1/39).

> Ponto HCAI: **a escolha das métricas é uma escolha de valores.** Medindo só a accuracy, não
> teríamos visto que algumas estratégias falham mais cancros.

### Como comparamos as estratégias de AL (contagem de vitórias)
- Para cada repetição (seed), calculamos a média da métrica **ao longo da curva toda** (de 20 a 270
  anotações), para cada estratégia.
- Contamos em quantas das 20 repetições a escolha ao acaso fica **acima** da estratégia.
- As duas começam das **mesmas 20 imagens** em cada repetição → comparação **emparelhada** (justa).
- Se fossem igualmente boas, o acaso ganharia ~10 de 20 vezes.
- (Já usámos o teste de Wilcoxon para isto; trocámo-lo por esta contagem, que é mais simples de
  explicar e mostra o mesmo.)

---

## 9. Active Learning

### A ideia
Anotar imagens custa tempo do médico. Em vez de anotar tudo, o **modelo escolhe** quais as imagens
que acha mais úteis e pede só essas ao especialista. Se escolher bem, chega ao mesmo resultado com
**menos anotações**.

*Pool-based AL* (o nosso caso): há um "pool" de imagens sem label; o modelo pontua-as com uma
**estratégia de consulta** (*query strategy*) e pede as labels das melhores.

### O ciclo (`run_active_learning` em `src/active_learning.py`)
1. Escolhem-se **20 imagens** do pool ao acaso (com as 3 classes) e anotam-se. São as **mesmas para
   todas as estratégias** na mesma seed.
2. Treina-se o modelo com essas imagens.
3. A estratégia escolhe as **10 imagens mais úteis** do resto do pool; o humano anota-as.
4. Retreina-se o modelo e avalia-se no teste.
5. Repete-se 3–4 **25 vezes** → até **270 imagens anotadas** (71% do pool).

Cada estratégia × 20 seeds = 80 corridas no total, em menos de 1 minuto.

### O modelo do AL: porquê uma regressão logística?
- Retreinar a CNN 25 vezes × 80 corridas era lento e ficava sem memória.
- Solução: cada imagem passa **uma só vez** pela MobileNetV2 **congelada** (pré-treinada no
  ImageNet), que a transforma num vetor de **1280 números** (*features*) que resumem o conteúdo da
  imagem (`extract_features`).
- O AL treina uma **regressão logística** sobre esses vetores: um classificador linear simples que
  treina em milissegundos.
- Isto é equivalente a treinar só a nova camada de saída da fase 1 da MobileNetV2.
- Detalhes: `StandardScaler` (normaliza as features), `class_weight="balanced"` (dá mais peso às
  classes com menos imagens), **C = 0.01** (regularização forte = modelo mais simples, menos
  overfitting; escolhido na validação).
- Referência: este classificador treinado com o **pool todo** (381) tem accuracy 0.785 e recall de
  maligno 0.718 (11 FN), perto da MobileNetV2 afinada.

### As 4 estratégias
| Estratégia | Escolhe as imagens em que… | De onde vem |
|---|---|---|
| **Random** (ao acaso) | — (ao acaso; é a referência) | nossa |
| **Least confidence** | a classe mais provável tem a probabilidade **mais baixa** | modAL (`uncertainty_sampling`) |
| **Margin** | as duas classes mais prováveis estão **mais próximas** | modAL (`margin_sampling`) |
| **Malignant** | a probabilidade de maligno está **mais perto de 50%** | **nossa** (`malignant_sampling`) |

**Exemplo** com três imagens (probabilidades normal / benigno / maligno):

| Imagem | Probabilidades | Least confidence (1 − máx) | Margin (1.º − 2.º) | Malignant (\|p_mal − 0.5\|) |
|---|---|---|---|---|
| A | 0.45 / 0.45 / 0.10 | 0.55 (alta incerteza) | 0.00 (muito indeciso) | 0.40 (longe) |
| B | 0.10 / 0.45 / 0.45 | 0.55 | 0.00 | 0.05 (**muito perto**) |
| C | 0.05 / 0.90 / 0.05 | 0.10 | 0.85 | 0.45 |

- Least confidence e margin acham A e B igualmente úteis. Mas A é uma dúvida **normal vs benigno**,
  que **não interessa** para detetar cancro.
- A malignant escolhe B: a dúvida "é cancro ou não?", que é a pergunta que interessa ao médico.

### Resultados
**Accuracy no teste (média de 20 seeds):**
| Estratégia | 50 | 100 | 150 | 200 | 270 |
|---|---|---|---|---|---|
| Random | **.671** | .733 | .763 | .787 | **.805** |
| Least confidence | .651 | .713 | .765 | .783 | .785 |
| Margin | .657 | **.734** | .769 | .785 | .783 |
| Malignant | .660 | .728 | **.770** | **.790** | .788 |

**Recall de maligno:**
| Estratégia | 50 | 100 | 150 | 200 | 270 |
|---|---|---|---|---|---|
| Random | .582 | **.651** | **.679** | **.700** | **.709** |
| Least confidence | .553 | .578 | .668 | .683 | .697 |
| Margin | **.603** | .614 | .656 | .665 | .694 |
| Malignant | .579 | .637 | .660 | .668 | .701 |

**Repetições (em 20) em que a escolha ao acaso ganha:**
| Estratégia | Accuracy | Recall de maligno |
|---|---|---|
| Least confidence | 16 | 13 |
| Margin | 14 | 12 |
| Malignant | 14 | 12 |

**Como ler isto:**
- **Nenhuma estratégia é melhor do que o acaso.** Margin e malignant ficam ao nível do acaso até às
  ~200 anotações (no máximo 0.7 pontos acima nalguns pontos, dentro da variação entre seeds). Com
  270 anotações, o acaso é o melhor (0.805 vs 0.783–0.788).
- **A least confidence deteta menos cancros no início:** com 100 anotações, 57.8% contra 65.1% do
  acaso (em média 16.5 cancros falhados de 39, contra 13.6).
- Com 270 anotações, todas convergem para recall ~0.70.
- **Cuidado (honestidade):** no recall de maligno ao longo da curva toda, 12–13 em 20 está perto do
  que o acaso daria (10) → evidência **fraca**. O que é claro: a diferença no início e a accuracy
  (16 em 20).
- Curiosidade: o acaso com 270 anotações (0.805) é **melhor** do que o classificador com as 381 do
  pool (0.785). Mostra que diferenças de poucos pontos dependem muito de *quais* imagens se anotam.

**Que imagens pede cada estratégia:**
- Random segue o pool: 16% normais, 53% benignas, 31% malignas.
- Least confidence e margin pedem **mais normais** (20–21%), menos benignas (48%), malignas iguais
  (32%) → gastam anotações na fronteira normal/benigno.
- Malignant pede **mais malignas** (38%), mas **não deteta mais cancros**.

### Porque é que o AL não ajuda aqui (3 razões plausíveis, não separadas)
1. As features pré-treinadas já separam bem as classes → uma amostra aleatória de poucas dezenas
   de imagens já é informativa.
2. As imagens onde o modelo está mais incerto são as mais **ambíguas**, e ambíguas são também as
   mais prováveis de ter **label errada** (o BUSI tem labels contraditórias). Concentrar as anotações
   nelas dá ao modelo um treino enviesado e com ruído.
3. Com 381 imagens no pool, há pouca margem para escolher: aos 270, 71% do pool está anotado em
   qualquer estratégia.

E porque é que a malignant também não ajuda: **perguntar sobre mais cancros não é o mesmo que
aprender a detetá-los**; as imagens com p(maligno) ≈ 50% voltam a ser as mais ambíguas.

### "Incerteza não é risco" (a ideia mais HCAI do AL)
As estratégias clássicas tratam as 3 classes por igual e gastam perguntas ao médico onde **um erro
custa pouco** (normal vs benigno). Num rastreio de cancro, isto é a troca errada.

### Comparação com o trabalho da UC (prof. Luís Macedo)
- Pinto et al. (2022): modAL + rede neuronal para diagnosticar 41 doenças a partir de sintomas, com
  menos dados.
- **Faria et al. (2022):** AL em previsão de ataque cardíaco (dados tabulares); todas as
  estratégias reduziram os dados necessários em **57–60%**. Esses ganhos **não se transferem** para
  o nosso problema → o benefício do AL depende muito dos dados, do modelo e do protocolo.
- Vieira et al. (2021): AL com ensemble para reconhecimento de objetos em carros autónomos.

---

## 10. O humano no ciclo (Human-in-the-Loop)

- **Human-in-the-Loop (HitL):** o humano participa no processo de aprendizagem/decisão. No AL, o
  médico é consultado **dentro** do ciclo de treino: o modelo aprende com a resposta dele.
- **No notebook 02**, com `HUMAN = True`, cada imagem escolhida é mostrada com a **sugestão do
  modelo**, e a pessoa escreve a classe correta (`ask_human`). O modelo é treinado com a label
  escrita, **mesmo que esteja errada**, como seria na realidade.
- **Nas experiências quantitativas**, o humano é **simulado** com a label do dataset (um
  especialista que nunca erra). Responder a 250 perguntas × 80 corridas não é viável para uma pessoa.
  Budd et al. (2021) notam que a maioria dos estudos faz o mesmo.
- **Riscos do humano real:**
  - cansaço, inexperiência;
  - **viés de automação** (*automation bias*, Goddard et al., 2012): tendência para aceitar a
    sugestão da máquina. Se o humano aceitar uma sugestão errada, o modelo aprende o próprio erro.
  - O AL manda ao humano precisamente as imagens **mais difíceis**, onde até especialistas
    discordam (as labels contraditórias do BUSI são prova disso).
- **Como mitigar:** esconder a sugestão até o humano responder, pedir um nível de confiança, ou
  mostrar onde o modelo olhou (Grad-CAM).
- **Human-in-the-Loop vs Machine-in-the-Loop** (slides do prof. Macedo): no AL, a máquina conduz a
  interação (escolhe o que perguntar), mas o humano tem a autoridade final sobre cada label.

---

## 11. Explicabilidade: Grad-CAM

### O que é
**Grad-CAM** (Selvaraju et al., 2017) produz um **mapa de calor** sobre a imagem que mostra que
zonas fizeram subir a pontuação de uma classe. Vermelho = zona que pesou mais na decisão.

### Como funciona na nossa rede (explicação simples)
1. A última camada convolucional da MobileNetV2 divide a imagem 128×128 numa **grelha de 4×4
   zonas** e descreve cada zona com **1280 features**.
2. A rede faz a média de cada feature em todas as zonas, e a camada final dá um **peso** a cada
   feature para cada classe.
3. O mapa de uma classe é, em cada zona, a **soma das features × esses pesos**. Zonas com valores
   altos = zonas que mais puxaram para essa classe.
4. O mapa 4×4 é ampliado para 128×128 para se sobrepor à imagem.

**Porque "Grad"?** Em geral o Grad-CAM calcula esses pesos com **gradientes** (quanto a pontuação
muda se uma feature mudar). Na nossa rede (média + uma só camada final), os gradientes são
exatamente os pesos da camada final (a menos de uma constante), por isso o mapa é igual e o código
nem precisa de calcular gradientes. (Isto chama-se CAM; o Grad-CAM é a generalização.)

**Limitação importante:** a grelha é **grosseira** (4×4). O mapa mostra a **zona** que o modelo
usou, não o contorno exato da lesão.

### Como verificamos se o modelo olha para a lesão
Para cada imagem de teste com lesão, comparamos o mapa com a **máscara desenhada pelos médicos**:
- **Mapa na lesão:** que fração da "massa" do mapa cai dentro da lesão.
- **Área da lesão:** que fração da imagem a lesão ocupa. É o que um mapa **que ignorasse a imagem**
  (uniforme) obteria → serve de comparação.

Modelo usado: MobileNetV2 da seed 1 (accuracy 0.799, recall de maligno 0.744, 10 FN).

| Imagens de teste | n | Mapa na lesão | Área da lesão |
|---|---|---|---|
| Malignas, detetadas | 29 | 35% | 18% |
| Malignas, falhadas | 10 | 24% | 13% |
| Benignas, certas | 63 | 15% | 7% |
| Benignas, erradas | 19 | 13% | 8% |

### O que concluímos
- Quando deteta um cancro, o modelo **olha sobretudo para a lesão**: 35% do mapa dentro de lesões
  que ocupam 18% da imagem, cerca do **dobro** do que um mapa ao acaso teria.
- Mas é grosseiro: mesmo nos cancros detetados, ~2/3 do mapa está fora da lesão.
- **Os 10 cancros falhados são de dois tipos:**
  1. **O modelo olhou para a lesão e achou-a benigna:** lesões grandes, ovais, de margens lisas
     (parecem benignas). → precisa de um modelo que veja melhor a margem (mais resolução,
     segmentação).
  2. **O modelo não olhou para a lesão** (sobretudo lesões pequenas): o mapa está no tecido ao lado
     ou acima. → precisa de um médico que verifique onde o modelo olhou.
- **O modelo engana-se com confiança:** 3 dos 10 cancros falhados são "benignos" com ≥97%, e 9 dos
  10 têm probabilidade de maligno < 25%. → **A confiança do modelo, sozinha, não chega para decidir
  o que o médico vê.**
- **Limites das explicações:** um mapa grosseiro que cobre a lesão pode fazer uma resposta errada
  parecer bem fundamentada.

---

## 12. O agente

Protótipo **num notebook** (`04_agente.ipynb`), não uma aplicação. Para cada ecografia:

1. Classifica a imagem com a MobileNetV2 (seed 1) e mostra o Grad-CAM.
2. Aplica a **regra de encaminhamento**: o caso vai para o médico se
   - o modelo prevê **maligno**, **ou**
   - a **confiança** (probabilidade da classe prevista) é **menor que τ**.
   Nesse caso, **nada** é comunicado ao paciente como resultado → médico **in the loop**.
3. Os restantes casos (normal ou benigno com confiança alta) são decididos pelo sistema, mas ficam
   num **registo (log)** que o médico revê e pode corrigir → médico **on the loop**.
4. Responde às perguntas do paciente com o chatbot.

### Human-in-the-loop vs Human-on-the-loop
- **In the loop:** o humano tem de intervir **antes** da decisão (casos suspeitos ou incertos).
- **On the loop:** o sistema decide, mas o humano **supervisiona** e pode corrigir depois (casos
  confiantes e não malignos). Implementado com `log_case`, `read_log`, `review_case` e o ficheiro
  `results/agent_log.csv`.

### Como se escolhe o limiar τ (`choose_threshold`)
- τ = o **menor** valor entre 0.50 e 0.99 (passos de 0.01) com que **pelo menos 95% dos malignos
  de validação** chegam ao médico.
- **Menor** τ → menos casos enviados ao médico (menos trabalho).
- Escolhido na **validação**, **nunca no teste**, para o teste continuar a medir a regra em imagens
  novas (honesto).
- Resultado: **τ = 0.88**.

### Resultados
| | Validação | Teste |
|---|---|---|
| Enviados ao médico | 71/127 (56%) | 82/149 (55%) |
| Cancros que chegam ao médico | 36/37 | 36/39 |
| Decisões automáticas erradas | 6/56 | 8/67 |

- No teste: dos 39 cancros, chegam ao médico os 29 que o modelo deteta + 7 dos 10 que falha (estes
  porque a confiança era < 0.88).
- Dos 67 casos decididos automaticamente, 8 estão errados: 5 benignos chamados normais e **3 cancros
  chamados benignos com 97–99% de confiança**. Estes 3 são os que a revisão do log pelo médico tem
  de apanhar.

### O dilema (trade-off)
Mais casos para o médico = mais trabalho, mas menos risco de **falsa tranquilização**. Com τ = 0.88,
o médico vê 55% dos casos e mesmo assim escapam 3 cancros. Dizer "benigno" a esses pacientes podia
atrasar o tratamento. Por isso:
- toda a mensagem automática diz que **não é um diagnóstico**;
- todos os casos ficam no log para revisão;
- um sistema real podia optar por não comunicar nada antes de o médico ver (perdendo o tempo que
  poupa).

### O que o paciente vê (`patient_message`)
- Caso enviado ao médico: "O resultado precisa de ser visto por um médico…" — **nunca a classe**.
- Caso automático: "O modelo classificou a imagem como {classe}, com confiança de {x}%… pode ser
  revisto pelo seu médico."
- Sempre: "Isto não é um diagnóstico."
- (Os textos finais são da issue #10, da colega.)

---

## 13. O chatbot (RAG + IAedu)

### O que é RAG
**RAG = Retrieval-Augmented Generation** (Lewis et al., 2020). Em vez de o modelo de linguagem
(LLM) responder "de memória" (e poder inventar), primeiro **procura** excertos relevantes em
documentos de confiança e responde **só com base neles**, citando-os. Respostas mais verificáveis e
menos invenção.

### Como funciona o nosso (`src/rag.py`)
1. **Documentos** médicos para pacientes ficam em `docs/rag/`, em ficheiros .md. São 19 textos
   em português, só de fontes públicas: SNS 24, DGS (incluindo a norma do rastreio), Liga Portuguesa
   Contra o Cancro, folhetos dos hospitais de São João e Santa Maria e a tradução da Sociedade
   Portuguesa de Radiologia das recomendações europeias (EUSOBI) sobre mamografia. Os PDFs foram
   passados a `.md` (o texto extraído vinha com colunas trocadas). A lista, com endereços e o que
   foi retirado de cada um (contactos, nomes de medicamentos, listas de perguntas sem resposta),
   está em `docs/rag/fontes.md`.
2. **Pedaços (chunks):** cada documento é dividido em pedaços de ~800 caracteres (juntam-se
   parágrafos seguidos; um parágrafo nunca é cortado).
3. **Embeddings:** cada pedaço é transformado num **vetor** que resume o seu significado, com um
   modelo multilingue (`paraphrase-multilingual-MiniLM-L12-v2`, via fastembed, corre no CPU,
   funciona em português). Textos com significados parecidos têm vetores próximos.
4. **Pesquisa:** a pergunta também vira vetor; procuram-se os **4 pedaços mais parecidos**
   (similaridade do **cosseno** = o quão alinhados estão os dois vetores, de −1 a 1).
5. **Mensagem para o LLM:** regras + resultado do caso + os 4 excertos numerados + a pergunta.
6. O LLM responde.

### As regras do chatbot
- Português de Portugal, frases curtas, sem jargão (ou explicando-o).
- Usar **só** os excertos e citar a fonte ([1], [2]…). Se não souber, dizer que não sabe e sugerir
  falar com o médico.
- **Nunca dar um diagnóstico**; a "tenho cancro?" **nunca responder sim nem não**: explicar que o
  modelo também erra, mesmo confiante, e que só o médico (e uma biópsia) pode responder.
- Não recomendar medicamentos nem tratamentos; não responder fora do tema.
- Tranquilizador mas honesto.
- Se o caso foi enviado ao médico, o chatbot **não recebe a classe** prevista
  (`case_for_chat`): só sabe que o exame vai ser revisto. Assim a classe não chega ao paciente antes
  de o médico ver.

### O episódio do "Não."
Na primeira versão, à pergunta "Isto quer dizer que tenho cancro?", com o modelo a prever benigno a
99%, o chatbot respondeu **"Não."**. Isto é **falsa tranquilização**: o modelo falha cancros com essa
confiança. Acrescentámos a regra de nunca responder sim ou não. É um bom exemplo para a
apresentação.

### O LLM: IAedu (`src/llm.py`)
- **IAedu** é a plataforma de IA generativa que a FCT disponibiliza ao ensino superior português
  (indicada pelo professor).
- Criámos um agente no IAedu; o código envia a mensagem por API (POST) e recebe a resposta aos
  bocados (*streaming*), que juntamos.
- As credenciais (`IAEDU_API_KEY`, `IAEDU_AGENT_ID`, `IAEDU_CHANNEL_ID`) ficam num ficheiro `.env`
  que **não vai para o git**.
- **Privacidade:** só enviamos a pergunta, os excertos e o resumo do caso. **Nenhum dado pessoal**.

### Riscos do chatbot
- Pode soar mais certo do que o modelo.
- Um LLM pode inventar conteúdo (*alucinação*).
- As mitigações (excertos citados, proibir diagnósticos e sim/não, esconder a classe nos casos
  encaminhados) **reduzem** os riscos mas **não os eliminam**.
- Ainda **não foi avaliado sistematicamente** (depende das perguntas de teste e da avaliação das
  issues #8 e #9).

### Como mostrar ao professor
No notebook 04, secção do chatbot: escolher um caso, fazer 2–3 perguntas (ex.: "o que é uma lesão
benigna?", "tenho cancro?", "que medicamento devo tomar?") e mostrar que cita as fontes, recusa o
diagnóstico e recusa medicação. Precisa do `.env` com a chave.

---

## 14. Ligação à IA Centrada no Humano

| Tema de HCAI | Onde aparece no projeto |
|---|---|
| **Human-in-the-Loop** | o médico anota no AL e o modelo aprende com ele; o agente manda casos suspeitos/incertos ao médico |
| **Human-on-the-Loop** | o médico revê o log das decisões automáticas |
| **Valores na avaliação** | recall de maligno e falsos negativos, não só accuracy; "incerteza não é risco" |
| **Confiança (trust)** | reportar honestamente os limites: cancros falhados, o leakage dos nossos primeiros resultados, a ausência de poupança no AL (Shneiderman, 2020) |
| **Explicabilidade** | Grad-CAM distingue "lesão mal julgada" de "lesão não vista", e testamos a explicação contra as máscaras dos médicos |
| **Controlo humano / automação** | Shneiderman: muita automação e muito controlo humano são compatíveis; o humano fica com as decisões importantes |
| **Viés de automação** | risco de o médico aceitar a sugestão do modelo; mitigações propostas |
| **Comunicação com o paciente / ética** | chatbot que nunca dá diagnóstico, cita fontes, não envia dados pessoais |

---

## 15. Limitações

(Saber dizê-las mostra solidez.)
1. **Teste pequeno:** 149 imagens, 39 malignas → 1 falso negativo = 2.6 pontos de recall. Todas as
   corridas usam **uma só divisão** dos dados.
2. O limiar de 0.95 para as repetidas foi escolhido **à mão**.
3. As configurações preliminares foram comparadas no **teste** da primeira divisão → a escolha do
   modelo usou parcialmente o teste. (Nas experiências de AL, nenhuma escolha usou o teste.)
4. O AL usa um classificador **linear sobre features congeladas**, não a rede afinada → as
   conclusões podem não se transferir para fine-tuning.
5. Rotações de 90° e 180° produzem imagens que um ecógrafo nunca produz.
6. Grad-CAM e agente usam **um só modelo** (seed 1).
7. O chatbot ainda não foi avaliado sistematicamente, nem com pacientes ou médicos.
8. Com 20 repetições, 12–13 em 20 está perto do acaso (10) → as diferenças no recall ao longo da
   curva são evidência fraca. Claras: a diferença no início e a accuracy.
9. Hardware limitado (4 GB GPU, 13 GB RAM) obrigou a imagens de 128×128 e a um AL com features.

### Trabalho futuro
1. Estratégias que pesem a incerteza pelo **custo de cada erro** e evitem imagens redundantes.
2. Mostrar o Grad-CAM em cada pergunta do AL; mapas mais finos com mais resolução.
3. AL com a rede afinada e várias divisões agrupadas.
4. Estudo com médicos e pacientes (tempo de anotação, efeito da sugestão, compreensão das
   mensagens).

---

## 16. O código: o que faz cada ficheiro

```
src/
  config.py          caminhos (dados, resultados, modelos, docs/rag)
  gpu.py             prepara a GPU, ativa o modo determinístico
  data.py            carregar imagens, detetar repetidas, dividir, balancear, máscaras das lesões
  exploration.py     gráficos do notebook 00
  model.py           CNN, MobileNetV2, fine-tuning, treino com early stopping, extração de features
  evaluation.py      métricas, matriz de confusão, curvas de aprendizagem, guardar JSON
  active_learning.py estratégias, classificador, ciclo de AL, pergunta ao humano
  explain.py         Grad-CAM e comparação com a lesão
  agent.py           regra de encaminhamento, limiar, mensagens ao paciente, log
  rag.py             documentos, pedaços, embeddings, pesquisa, regras do chatbot
  llm.py             ligação à API do IAedu
notebooks/           00 a 04 (ver secção seguinte)
results/             passive_learning.json, active_learning.json (métricas guardadas)
models/              mobilenet.keras (modelo treinado; não vai para o git)
report/              relatório LaTeX (formato IJCAI) e figures/make_figures.py
docs/rag/            documentos médicos para o chatbot
.env                 chave do IAedu (não vai para o git)
```

**Funções principais:**
- `find_duplicates` — agrupa ecografias repetidas (correlação > 0.95 em 32×32).
- `load_dataset` — divide por grupos (StratifiedGroupKFold), balanceia só o treino.
  Opções: `balance=False` (sem augmentation, para o AL), `unique_train=True` (1 imagem por
  ecografia), `test_names=True` (devolve os nomes do teste, para as máscaras).
- `build_cnn`, `build_mobilenet`, `unfreeze_top`, `fit`, `extract_features`.
- `compute_metrics` — accuracy, recall de maligno, FN.
- `run_active_learning` — o ciclo de AL; `STRATEGIES` — as 4 estratégias.
- `GradCAM`, `lesion_overlap`.
- `to_doctor`, `choose_threshold`, `decide`, `patient_message`, `case_for_chat`, `log_case`,
  `review_case`.
- `Retriever`, `answer`, `IAedu.ask`.

**Reprodutibilidade:** código partilhado em `src/`, resultados guardados em JSON, figuras do relatório
geradas por um script a partir desses JSON, README com instruções, treino determinístico.

---

## 17. Os notebooks, passo a passo

Correm por ordem, localmente na GPU.

- **00 — Exploração dos dados:** contagens por classe, imagens repetidas e as 19 contraditórias,
  exemplos, balanceamento, tamanhos das imagens e memória (porquê 128×128).
- **01 — Passive learning:** treina a CNN e a MobileNetV2 com 5 seeds; tabela de métricas; matrizes
  de confusão; mostra os cancros falhados; guarda o modelo da seed 1 (`models/mobilenet.keras`) e os
  resultados (`results/passive_learning.json`).
- **02 — Active learning:** extrai as features uma vez; corre as 4 estratégias × 20 seeds; curvas,
  tabelas, contagem de vitórias, que imagens cada estratégia pede; secção interativa com
  `HUMAN = True` para anotar à mão.
- **03 — Explicabilidade:** Grad-CAM nos cancros detetados e falhados, com o contorno da lesão;
  tabela de sobreposição com as máscaras; confiança nos erros.
- **04 — Agente:** escolhe τ na validação; aplica no teste; mostra casos com Grad-CAM e a mensagem
  ao paciente; log e revisão pelo médico; chatbot (precisa do `.env`).

---

## 18. Folha de números

| O quê | Valor |
|---|---|
| Ficheiros no BUSI | 780 (133 normal, 437 benigno, 210 maligno) |
| Ecografias diferentes | 601 |
| Excluídas (labels contraditórias) | 19 → ficam 761 |
| Treino / validação / teste | 485 / 127 / 149 (teste: 39 malignos) |
| Treino balanceado | 1554 |
| Pool do AL | 381 |
| Tamanho das imagens | 128×128, cinzento |
| MobileNetV2 (teste) | accuracy 0.804, recall maligno 0.697, ~11.8 FN |
| CNN (teste) | accuracy 0.609, recall maligno 0.718 |
| MobileNetV2 com leakage | accuracy 0.840, recall maligno 0.802 |
| Poupança aparente do AL com leakage | 24% (190 vs 250 anotações) |
| AL: início / passo / rondas | 20 / 10 / 25 → 270 anotações |
| AL: repetições | 20 seeds × 4 estratégias |
| Classificador do AL | regressão logística, C = 0.01, 1280 features |
| AL com 270: accuracy | random 0.805; outras 0.783–0.788 |
| AL com 100: recall maligno | random 65.1%; least confidence 57.8% |
| Vitórias do acaso (accuracy) | 16/20 (LC), 14/20 (margin, malignant) |
| Vitórias do acaso (recall) | 13/20 (LC), 12/20 (margin, malignant) |
| Pool todo (381) | accuracy 0.785, recall 0.718 |
| Grad-CAM: cancros detetados | 35% do mapa na lesão (lesão = 18% da imagem) |
| Cancros falhados com ≥97% confiança | 3 de 10 |
| Limiar do agente τ | 0.88 (escolhido na validação, alvo 95%) |
| Agente no teste | 55% ao médico; 36/39 cancros chegam; 8/67 automáticos errados |
| Hardware | RTX 3050 (4 GB), 13 GB RAM |

---

## 19. Perguntas prováveis e respostas

**"Então o Active Learning não funcionou?"**
Não funcionou *neste* problema, e isso é o resultado. O contributo é o **protocolo justo** (divisão
por ecografia, baseline aleatória, 20 repetições emparelhadas, recall de maligno além da accuracy),
que mostra que ganhos de AL publicados podem ser artefactos. Os 57–60% de Faria et al. em dados
tabulares não se transferem.

**"Como sabem que as imagens são mesmo repetidas?"**
Correlação > 0.95 entre as imagens reduzidas a 32×32. O limiar é manual, mas no notebook 00
mostramos exemplos dos grupos lado a lado, e são visivelmente a mesma ecografia. Uma delas
(`benign (433)` = `malignant (145)`) é até byte a byte igual.

**"Porque excluíram as 19 imagens?"**
São a mesma ecografia com classes diferentes. Não sabemos qual está certa, e treinar com labels
contraditórias só confunde o modelo.

**"As diferenças entre estratégias são reais?"**
Com 20 repetições, 12–13 em 20 está perto do acaso (10): é evidência fraca, e dizemo-lo nas
limitações. Claros são: o acaso ganha na accuracy (16 em 20 contra least confidence) e a diferença
no início (58% vs 65% dos cancros com 100 anotações).

**"Porque usaram uma regressão logística no AL e não a CNN?"**
Retreinar a CNN 25 vezes × 80 corridas era lento e ficava sem memória. A regressão logística sobre
features da MobileNetV2 congelada treina em milissegundos e é equivalente a treinar só a camada de
saída. Com o pool todo chega a 0.785, perto da rede afinada (0.804). Limitação: pode não se
transferir para fine-tuning.

**"Porque 128×128?"**
Com 224 ficava sem RAM, e não era melhor. 128×128 ocupa ~120 MB.

**"O que é o Grad-CAM e porque confiar nele?"**
Mapa das zonas que mais puxaram para a classe. Não confiamos cegamente: **testámo-lo** contra as
lesões desenhadas pelos médicos. Mas é grosseiro (4×4) e pode fazer uma resposta errada parecer bem
fundamentada.

**"Porque é que o agente manda todos os malignos ao médico?"**
Porque um falso negativo é o erro mais grave e porque o modelo falha cancros com muita confiança
(a confiança sozinha não chega). Um "maligno" nunca deve chegar ao paciente sem o médico.

**"Porque 0.88?"**
É o menor limiar com que ≥95% dos malignos de **validação** chegam ao médico. Menor limiar = menos
trabalho para o médico. Escolhido na validação, não no teste.

**"E os 3 cancros que escapam ao agente?"**
São decididos como benignos com 97–99%. Por isso há o log que o médico revê (on the loop), e a
mensagem diz sempre que não é um diagnóstico. Um sistema real podia não comunicar nada antes do
médico.

**"O chatbot pode inventar?"**
Pode; os LLMs alucinam. Mitigámos com RAG (só excertos, com citação), regras explícitas (nunca
diagnóstico, nunca sim/não, nunca medicação) e escondendo a classe dos casos encaminhados. Reduz,
não elimina. Ainda falta a avaliação sistemática.

**"E a privacidade?"**
Só vai para o IAedu a pergunta, os excertos e o resumo do caso; nenhum dado pessoal. A chave da API
nunca vai para o git.

**"O humano é real ou simulado?"**
Nas experiências, simulado (label do dataset, especialista perfeito), como na maioria dos estudos.
Mas o modo `HUMAN = True` do notebook 02 permite anotar à mão, e o modelo aprende com a resposta,
mesmo errada.

**"Qual é o maior risco do Human-in-the-Loop aqui?"**
Viés de automação: o médico aceitar a sugestão do modelo. E o AL manda-lhe precisamente as imagens
mais difíceis. Mitigação: esconder a sugestão, pedir confiança, mostrar o Grad-CAM.

**"Os resultados são reprodutíveis?"**
Sim: treino determinístico (mesma seed → mesmo modelo), resultados em JSON, figuras geradas por
script, README com instruções.

**"O que fariam a seguir?"**
Estratégias sensíveis ao custo do erro; Grad-CAM nas perguntas do AL; AL com a rede afinada e várias
divisões; estudo com médicos e pacientes.

---

## 20. Glossário

| Termo | Significado |
|---|---|
| **Accuracy** | % de imagens bem classificadas |
| **Active Learning (AL)** | o modelo escolhe que imagens o especialista deve anotar |
| **Adam** | algoritmo que ajusta os pesos durante o treino |
| **Augmentation** | criar mais imagens de treino com rotações/reflexões |
| **Automation bias** | tendência humana para aceitar a sugestão da máquina |
| **Batch / mini-batch** | grupo de imagens usado em cada passo de treino (32) |
| **Batch normalization** | normalização entre camadas, estabiliza o treino |
| **BUSI** | o dataset de ecografias mamárias |
| **CAM / Grad-CAM** | mapa das zonas da imagem que pesaram na decisão |
| **Chunk** | pedaço de um documento (~800 caracteres) no RAG |
| **CNN** | rede neuronal convolucional, própria para imagens |
| **Confiança** | probabilidade da classe prevista |
| **Data leakage** | o treino "vê" informação do teste → resultados inflacionados |
| **Dropout** | desligar neurónios ao acaso no treino, contra o overfitting |
| **Early stopping** | parar o treino quando a validação deixa de melhorar |
| **Embedding** | vetor que representa o significado de um texto |
| **Época** | uma passagem por todo o treino |
| **Falso negativo (FN)** | cancro classificado como benigno/normal |
| **Falso positivo** | não-cancro classificado como maligno |
| **Features** | números que resumem a imagem (aqui, 1280 da MobileNetV2) |
| **Fine-tuning** | afinar as últimas camadas de uma rede pré-treinada |
| **Human-in-the-loop** | o humano intervém antes/durante a decisão ou aprendizagem |
| **Human-on-the-loop** | o sistema decide, o humano supervisiona e corrige |
| **IAedu** | plataforma de IA generativa da FCT, o nosso LLM |
| **ImageNet** | grande dataset de fotografias usado para pré-treinar redes |
| **Learning rate** | tamanho de cada ajuste dos pesos |
| **Least confidence** | estratégia: escolhe onde a classe mais provável tem menor probabilidade |
| **LLM** | modelo de linguagem grande (gera texto) |
| **Logistic regression** | classificador linear simples, usado no AL |
| **Loss** | medida do erro que o treino minimiza |
| **Margin** | estratégia: escolhe onde as duas classes mais prováveis estão mais próximas |
| **Matriz de confusão** | tabela classe real × classe prevista |
| **MobileNetV2** | CNN pequena pré-treinada no ImageNet |
| **modAL** | biblioteca Python de Active Learning |
| **Overfitting** | o modelo decora o treino e generaliza mal |
| **Passive learning** | treinar com todas as labels, sem escolher |
| **Pool** | imagens sem label de onde o AL escolhe |
| **Query strategy** | regra que decide que imagens pedir ao humano |
| **RAG** | o LLM responde com base em excertos de documentos recuperados |
| **Recall de maligno** | % dos cancros que o modelo deteta |
| **ReLU** | função de ativação max(0, x) |
| **Seed** | número que fixa o acaso, para reprodutibilidade |
| **Similaridade do cosseno** | quão parecidos são dois vetores (−1 a 1) |
| **Softmax** | transforma as saídas em probabilidades que somam 1 |
| **StratifiedGroupKFold** | divisão que mantém proporções das classes e grupos juntos |
| **Transfer learning** | reaproveitar uma rede treinada noutro problema |
| **τ (tau)** | limiar de confiança do agente (0.88) |
| **Validação** | conjunto para afinar decisões de treino, nunca para o resultado final |

---

## 21. Sugestão de apresentação

1. **Problema clínico** — o custo assimétrico dos erros (falso negativo é o pior).
2. **Descoberta das imagens repetidas** — 780 ficheiros, 601 ecografias; 0.840 → 0.804; os 24%
   desaparecem. Mostrar um grupo de imagens repetidas.
3. **Resultado do AL** — curvas; nenhuma estratégia bate o acaso; contagem de vitórias.
4. **"Incerteza não é risco"** — as estratégias gastam perguntas na fronteira normal/benigno; least
   confidence deteta menos cancros no início; a nossa "malignant" pergunta mais sobre cancros mas não
   deteta mais.
5. **Papel do humano** — HitL real no notebook; viés de automação; o modelo falha 12 em 39.
6. **Grad-CAM** — dois tipos de cancros falhados; confiança alta nos erros.
7. **Agente e chatbot** — regra de encaminhamento (τ = 0.88, 36/39); log on-the-loop; o episódio do
   "Não.".
8. **Limitações** e trabalho futuro.

**História:** *"O que parecia um ganho era um artefacto, e o que o modelo pergunta ao médico
importa."*

"""Ciclo de Active Learning com Human-in-the-Loop (modAL).

O modelo é um classificador (regressão logística) treinado sobre as features da MobileNetV2 congelada
(ver extract_features em src/model.py). Como é rápido de treinar, pode ser treinado de novo em cada
iteração e as experiências podem repetir-se com várias seeds.
"""

import time
import warnings

import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from modAL.utils.selection import multi_argmax
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASSES
from .evaluation import compute_metrics

# O modAL usa um nome antigo de um argumento do scikit-learn; o aviso não afeta os resultados
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)


def random_sampling(classifier, X_pool, n_instances=1, rng=None):
    """Estratégia de referência: escolhe imagens ao acaso, sem olhar para o modelo."""
    rng = rng if rng is not None else np.random.default_rng()
    return rng.choice(len(X_pool), size=n_instances, replace=False)


def malignant_sampling(classifier, X_pool, n_instances=1):
    """Estratégia nossa, focada no objetivo do projeto: escolhe as imagens em que o modelo está mais
    indeciso sobre se são malignas ou não, ou seja, com probabilidade de maligno mais perto de 0.5.

    As estratégias do modAL olham para as 3 classes por igual, e muitas das imagens que escolhem estão
    na fronteira normal/benigno, que não interessa para detetar cancro.
    """
    p_malignant = classifier.predict_proba(X_pool)[:, CLASSES.index("malignant")]
    return multi_argmax(-np.abs(p_malignant - 0.5), n_instances=n_instances)


STRATEGIES = {
    "random": random_sampling,
    "uncertainty": uncertainty_sampling,
    "entropy": entropy_sampling,
    "margin": margin_sampling,
    "malignant": malignant_sampling,
}


def make_classifier(C=0.01):
    """Regressão logística sobre as features normalizadas.

    C controla a regularização (menor = modelo mais simples, menos overfitting).
    class_weight="balanced" dá mais peso às classes com menos imagens anotadas.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def ask_human(image, suggestion):
    """Mostra a imagem e pede ao humano a classe correta."""
    from IPython.display import display

    display(Image.fromarray((image * 255).astype(np.uint8), mode="L"))
    time.sleep(0.1)
    options = ", ".join(f"{i} = {name}" for i, name in enumerate(CLASSES))
    return int(input(f"Classifique a imagem ({options}).\nSugestão do modelo: {suggestion} ({CLASSES[suggestion]}): "))


def run_active_learning(strategy, f_train, y_train, f_test, y_test, n_initial=20, n_instances=10, n_queries=25,
                        C=0.01, seed=0, human=False, images=None):
    """Corre um ciclo de Active Learning e devolve as métricas no teste depois de cada iteração.

    f_train / f_test: features da MobileNetV2 (uma linha por imagem).
    n_initial: imagens anotadas no início, escolhidas ao acaso (as mesmas para todas as estratégias com a
               mesma seed, para a comparação ser justa).
    n_instances: imagens pedidas ao humano em cada iteração; n_queries: nº de iterações.
    human=True pede a label a uma pessoa (precisa de `images`, as imagens de treino, para as mostrar);
    human=False simula o humano com a label verdadeira do dataset.
    Em ambos os casos o modelo aprende com a label dada pelo "humano".
    """
    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(len(f_train), size=n_initial, replace=False)
    pool_idx = np.setdiff1d(np.arange(len(f_train)), initial_idx)   # índices das imagens ainda sem label

    query_rng = np.random.default_rng(seed + 1000)   # só usado pela estratégia random
    query_strategy = STRATEGIES[strategy]
    if strategy == "random":
        query_strategy = lambda clf, X, n_instances=1: random_sampling(clf, X, n_instances, query_rng)

    learner = ActiveLearner(estimator=make_classifier(C), query_strategy=query_strategy,
                            X_training=f_train[initial_idx], y_training=y_train[initial_idx])

    history = {"n_labeled": [], "queried": []}

    def record():
        history["n_labeled"].append(len(learner.y_training))
        for name, value in compute_metrics(y_test, learner.predict(f_test)).items():
            history.setdefault(name, []).append(value)

    record()
    for _ in range(n_queries):
        result = learner.query(f_train[pool_idx], n_instances=n_instances)
        query = np.asarray(result[0] if isinstance(result, tuple) else result)   # posições dentro da pool
        chosen = pool_idx[query]                                                 # índices no conjunto de treino

        if human:
            suggestions = learner.predict(f_train[chosen])
            labels = np.array([ask_human(images[i], int(s)) for i, s in zip(chosen, suggestions)])
        else:
            labels = y_train[chosen]

        learner.teach(f_train[chosen], labels)   # junta as imagens ao conjunto anotado e volta a treinar
        pool_idx = np.delete(pool_idx, query)
        history["queried"].extend(int(i) for i in chosen)
        record()

    history["y_pred"] = learner.predict(f_test).tolist()
    return history

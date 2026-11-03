"""Ciclo de Active Learning com Human-in-the-Loop (modAL + scikeras)."""

import time

import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from PIL import Image
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import to_categorical

from .data import CLASSES
from .model import build_cnn

STRATEGIES = {
    "uncertainty": uncertainty_sampling,
    "entropy": entropy_sampling,
    "margin": margin_sampling,
}


def ask_human(image, suggestion):
    """Mostra a imagem e pede ao humano a classe correta."""
    from IPython.display import display

    display(Image.fromarray((image * 255).astype(np.uint8), mode="L"))
    time.sleep(0.1)
    options = ", ".join(f"{i} = {name}" for i, name in enumerate(CLASSES))
    return int(input(f"Classifique a imagem ({options}).\nSugestão do modelo: {suggestion} ({CLASSES[suggestion]}): "))


def run_active_learning(strategy, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=32, epochs=10,
                        n_initial=1000, n_queries=10, human=False, seed=None):
    """Treina uma CNN com Active Learning e devolve o histórico de métricas e as previsões no teste.

    human=True pede a label a uma pessoa em cada iteração; human=False simula o humano com a label
    verdadeira do dataset (útil para correr as experiências sem intervenção manual).
    Em ambos os casos o modelo é treinado com a label dada pelo "humano".
    O conjunto de validação (imagens que nunca entram no treino) acompanha cada treino do modelo.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(CLASSES)
    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=n_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=n_classes)

    size = x_train.shape[1]
    keras_clf = KerasClassifier(model=build_cnn, model__size=size, model__n_classes=n_classes,
                                epochs=epochs, batch_size=batch_size, warm_start=True, verbose=0,
                                fit__validation_data=(x_val, y_val_one_hot))

    # Usar alguns dados de treino como conjunto inicial já anotado
    initial_idx = rng.choice(len(x_train), size=n_initial, replace=False)
    x_labeled, y_labeled = x_train[initial_idx], y_train_one_hot[initial_idx]
    x_pool, y_pool = np.delete(x_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)

    learner = ActiveLearner(estimator=keras_clf, query_strategy=STRATEGIES[strategy],
                            X_training=x_labeled, y_training=y_labeled)

    history = {"accuracy_train": [], "loss_train": [], "accuracy_test": [], "loss_test": []}

    # Loop de Active Learning
    for _ in range(n_queries):
        query_idx, _ = learner.query(x_pool)
        query_image = x_pool[query_idx]

        if human:
            suggestion = int(np.argmax(learner.estimator.model_.predict(query_image, verbose=0)))
            label = ask_human(query_image[0], suggestion)
        else:
            label = int(y_pool[query_idx][0])
        label = to_categorical([label], num_classes=n_classes)

        learner.teach(query_image, label)
        x_labeled = np.append(x_labeled, query_image, axis=0)
        y_labeled = np.append(y_labeled, label, axis=0)

        # Loss e accuracy nos dados anotados até agora e nos dados de teste
        loss, accuracy = learner.estimator.model_.evaluate(x_labeled, y_labeled, verbose=0)
        history["accuracy_train"].append(accuracy)
        history["loss_train"].append(loss)
        loss, accuracy = learner.estimator.model_.evaluate(x_test, y_test_one_hot, verbose=0)
        history["accuracy_test"].append(accuracy)
        history["loss_test"].append(loss)

        x_pool = np.delete(x_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

    y_pred = np.argmax(learner.estimator.model_.predict(x_test, verbose=0), axis=1)
    return history, y_pred

"""Carregamento, balanceamento e divisão do dataset de ecografias mamárias."""

import hashlib
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["normal", "benign", "malignant"]  # índice = label numérica

# Transformações usadas para balancear cada classe (k = nº de rotações de 90º, flip = reflexão vertical).
# normal: 133 -> x6, maligno: 210 -> x4, benigno: 437 -> x2
AUGMENTATIONS = {
    "normal":    [(2, False), (1, False), (0, True), (2, True), (1, True)],
    "malignant": [(2, False), (0, True), (2, True)],
    "benign":    [(0, True)],
}


def find_duplicates(path):
    """Nomes dos ficheiros que aparecem mais do que uma vez no dataset (conteúdo igual).

    O dataset tem pelo menos um caso: "benign (433).png" e "malignant (145).png" são a mesma imagem
    com labels diferentes. Como não se sabe qual está certa, as cópias são todas excluídas.
    """
    groups = {}
    for name in sorted(os.listdir(path)):
        with open(os.path.join(path, name), "rb") as f:
            groups.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(name)
    return {name for names in groups.values() if len(names) > 1 for name in names}


def load_images(path, size=224):
    """Lê as imagens em escala de cinzento, redimensionadas e normalizadas para [0, 1].

    As classes vêm do nome do ficheiro (ex.: "benign (12).png"). As imagens duplicadas são excluídas.
    """
    duplicates = find_duplicates(path)
    names = [name for name in sorted(os.listdir(path)) if name not in duplicates]
    images = np.array([
        np.asarray(Image.open(os.path.join(path, name)).convert("L").resize((size, size)), dtype=np.float32) / 255.0
        for name in names
    ])
    labels = np.array([CLASSES.index(name.split(" ")[0]) for name in names])
    return images, labels


def augment(images, labels):
    """Balanceia as classes acrescentando rotações e reflexões das imagens existentes."""
    new_images, new_labels = [images], [labels]
    for class_name, transforms in AUGMENTATIONS.items():
        class_images = images[labels == CLASSES.index(class_name)]
        for k, flip in transforms:
            transformed = np.rot90(class_images, k=k, axes=(1, 2))
            if flip:
                transformed = transformed[:, ::-1, :]
            new_images.append(transformed)
            new_labels.append(np.full(len(class_images), CLASSES.index(class_name)))
    return np.concatenate(new_images), np.concatenate(new_labels)


def load_dataset(path, size=224, val_size=0.15, test_size=0.2, seed=1, balance=True):
    """Devolve x_train, x_val, x_test, y_train, y_val, y_test (labels numéricas).

    As imagens originais são divididas em treino (65%), validação (15%) e teste (20%), de forma
    estratificada, ANTES do balanceamento. Assim, nenhuma imagem de validação ou de teste (nem uma
    versão rodada dela) é usada no treino (data leakage). Só o treino é balanceado; a validação e o
    teste ficam com as imagens originais e a distribuição real das classes.

    balance=False devolve o treino só com as imagens originais (usado no Active Learning, onde o humano
    anota imagens reais e não cópias rodadas).
    """
    images, labels = load_images(path, size)

    x_rest, x_test, y_rest, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=val_size / (1 - test_size), stratify=y_rest, random_state=seed)

    if not balance:
        return x_train, x_val, x_test, y_train, y_val, y_test

    # Balancear e baralhar só o treino (o augment junta as cópias por classe no fim)
    x_train, y_train = augment(x_train, y_train)
    order = np.random.default_rng(seed).permutation(len(x_train))
    return x_train[order], x_val, x_test, y_train[order], y_val, y_test


def summarize(x_train, x_val, x_test, y_train, y_val, y_test):
    """Mostra o tamanho, a memória ocupada e a distribuição das classes de cada conjunto."""
    print(f"Treino: {x_train.shape}, validação: {x_val.shape}, teste: {x_test.shape}")
    print(f"Memória das imagens: {(x_train.nbytes + x_val.nbytes + x_test.nbytes) / 1e6:.0f} MB")
    for split, y in [("Treino", y_train), ("Validação", y_val), ("Teste", y_test)]:
        print(f"{split}:", {name: int(np.sum(y == i)) for i, name in enumerate(CLASSES)})

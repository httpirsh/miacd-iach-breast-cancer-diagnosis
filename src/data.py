"""Carregamento, balanceamento e divisão do dataset de ecografias mamárias."""

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


def load_images(path, size=224):
    """Lê as imagens em escala de cinzento, redimensionadas e normalizadas para [0, 1].

    As classes vêm do nome do ficheiro (ex.: "benign (12).png").
    """
    names = sorted(os.listdir(path))
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


def load_dataset(path, size=224, test_size=0.2, seed=1):
    """Devolve x_train, x_test, y_train, y_test (labels numéricas).

    A divisão treino/teste é feita ANTES do balanceamento, para que versões rodadas
    da mesma imagem não fiquem em treino e teste ao mesmo tempo (data leakage).
    O conjunto de teste fica com as imagens originais e a distribuição real das classes.
    """
    images, labels = load_images(path, size)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=seed)
    x_train, y_train = augment(x_train, y_train)
    return x_train, x_test, y_train, y_test


def summarize(x_train, x_test, y_train, y_test):
    """Mostra o tamanho, a memória ocupada e a distribuição das classes de cada conjunto."""
    print(f"Treino: {x_train.shape}, teste: {x_test.shape}")
    print(f"Memória das imagens: {(x_train.nbytes + x_test.nbytes) / 1e6:.0f} MB")
    for split, y in [("Treino", y_train), ("Teste", y_test)]:
        print(f"{split}:", {name: int(np.sum(y == i)) for i, name in enumerate(CLASSES)})

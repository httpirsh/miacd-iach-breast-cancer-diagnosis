"""Carregamento, balanceamento e divisão do dataset de ecografias mamárias."""

import hashlib
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["normal", "benign", "malignant"]  # índice = label numérica

# Transformações usadas para balancear cada classe (k = nº de rotações de 90º, flip = reflexão vertical).
# normal: 133 -> x6, maligno: 210 -> x4, benigno: 437 -> x2
AUGMENTATIONS = {
    "normal":    [(2, False), (1, False), (0, True), (2, True), (1, True)],
    "malignant": [(2, False), (0, True), (2, True)],
    "benign":    [(0, True)],
}


def label_from_name(name):
    """Classe de uma imagem, tirada do nome do ficheiro (ex.: "benign (12).png" -> "benign")."""
    return name.split(" ")[0]


def find_duplicates(path, threshold=0.95):
    """Agrupa as imagens repetidas do dataset: a mesma ecografia guardada em ficheiros diferentes.

    O dataset tem muitas imagens quase iguais (a mesma ecografia com um recorte ligeiramente diferente),
    por isso comparar os ficheiros byte a byte não chega. Cada imagem é reduzida a 32×32 píxeis e
    normalizada; duas imagens são consideradas a mesma se a correlação entre os píxeis for > threshold.

    Devolve:
    - groups: {nome do ficheiro: nº do grupo}; imagens repetidas ficam no mesmo grupo;
    - conflicts: nomes das imagens cujo grupo tem labels diferentes (ex.: "benign (433).png" e
      "malignant (145).png" são a mesma imagem). Como não se sabe qual label está certa, são excluídas.
    """
    names = sorted(os.listdir(path))
    pixels = np.array([np.asarray(Image.open(os.path.join(path, name)).convert("L").resize((32, 32)),
                                  dtype=np.float32).ravel() for name in names])
    pixels = (pixels - pixels.mean(axis=1, keepdims=True)) / (pixels.std(axis=1, keepdims=True) + 1e-6)
    similarity = pixels @ pixels.T / pixels.shape[1]   # correlação entre cada par de imagens

    # Juntar em grupos (se A ~ B e B ~ C, então A, B e C ficam no mesmo grupo)
    parent = list(range(len(names)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(similarity > threshold, k=1))):
        parent[root(i)] = root(j)

    groups = {name: root(i) for i, name in enumerate(names)}
    labels_per_group = {}
    for name, group in groups.items():
        labels_per_group.setdefault(group, set()).add(label_from_name(name))
    conflicts = {name for name, group in groups.items() if len(labels_per_group[group]) > 1}
    return groups, conflicts


def load_images(path, size=224):
    """Lê as imagens em escala de cinzento, redimensionadas e normalizadas para [0, 1].

    As classes vêm do nome do ficheiro (ex.: "benign (12).png"). As imagens repetidas com labels
    diferentes são excluídas. Devolve também o grupo de cada imagem (imagens repetidas têm o mesmo grupo)
    e o nome do ficheiro.
    """
    groups, conflicts = find_duplicates(path)
    names = [name for name in sorted(os.listdir(path)) if name not in conflicts]
    images = np.array([
        np.asarray(Image.open(os.path.join(path, name)).convert("L").resize((size, size)), dtype=np.float32) / 255.0
        for name in names
    ])
    labels = np.array([CLASSES.index(label_from_name(name)) for name in names])
    return images, labels, np.array([groups[name] for name in names]), np.array(names)


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


def load_dataset(path, size=224, seed=1, balance=True, unique_train=False, test_names=False):
    """Devolve x_train, x_val, x_test, y_train, y_val, y_test (labels numéricas).

    As imagens originais são divididas em treino (~64%), validação (~17%) e teste (~20%), de forma
    estratificada e por grupos (as cópias da mesma ecografia ficam sempre no mesmo conjunto), ANTES do
    balanceamento. Assim, nenhuma imagem de validação ou de teste (nem uma cópia ou versão rodada dela)
    é usada no treino (data leakage). Só o treino é balanceado; a validação e o
    teste ficam com as imagens originais e a distribuição real das classes.

    balance=False devolve o treino só com as imagens originais (usado no Active Learning, onde o humano
    anota imagens reais e não cópias rodadas).
    unique_train=True deixa no treino só uma imagem de cada grupo de repetidas (no Active Learning, o
    humano não deve ter de anotar a mesma ecografia duas vezes).
    test_names=True devolve também os nomes dos ficheiros do teste (para ir buscar as máscaras das lesões).
    """
    images, labels, groups, names = load_images(path, size)

    # Divisão por grupos: todas as cópias de uma ecografia ficam no mesmo conjunto.
    # StratifiedGroupKFold com 5 partes: uma parte (20%) para teste; depois, do resto, outra vez 1 de 5 partes
    # (⅕ de 80%, ~17% do total) para validação; fica ~64% para treino.
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, test = next(folds.split(images, labels, groups))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, val = next(folds.split(images[rest], labels[rest], groups[rest]))
    train, val = rest[train], rest[val]
    if unique_train:
        _, first = np.unique(groups[train], return_index=True)
        train = train[np.sort(first)]
    x_train, x_val, x_test = images[train], images[val], images[test]
    y_train, y_val, y_test = labels[train], labels[val], labels[test]

    extra = (names[test],) if test_names else ()

    if not balance:
        return (x_train, x_val, x_test, y_train, y_val, y_test) + extra

    # Balancear e baralhar só o treino (o augment junta as cópias por classe no fim)
    x_train, y_train = augment(x_train, y_train)
    order = np.random.default_rng(seed).permutation(len(x_train))
    return (x_train[order], x_val, x_test, y_train[order], y_val, y_test) + extra


def load_lesion_mask(path, name, size=224):
    """Máscara da lesão desenhada pelos médicos (True = lesão), redimensionada como a imagem.

    As anotações estão em ds/ann/<nome>.json (formato Supervisely): cada lesão é um bitmap (PNG comprimido
    com zlib, em base64) com a posição do canto superior esquerdo. As imagens normais não têm lesão.
    """
    import base64
    import io
    import json
    import zlib

    with open(os.path.join(os.path.dirname(path), "ann", name + ".json")) as f:
        ann = json.load(f)
    mask = np.zeros((ann["size"]["height"], ann["size"]["width"]), dtype=bool)
    for obj in ann["objects"]:
        bitmap = np.asarray(Image.open(io.BytesIO(zlib.decompress(base64.b64decode(obj["bitmap"]["data"])))))
        bitmap = bitmap[..., -1] > 0 if bitmap.ndim == 3 else bitmap > 0   # canal alpha = lesão
        x, y = obj["bitmap"]["origin"]
        h, w = bitmap.shape
        mask[y:y + h, x:x + w] |= bitmap[:mask.shape[0] - y, :mask.shape[1] - x]
    return np.asarray(Image.fromarray(mask).resize((size, size), Image.NEAREST))


def summarize(x_train, x_val, x_test, y_train, y_val, y_test):
    """Mostra o tamanho, a memória ocupada e a distribuição das classes de cada conjunto."""
    print(f"Treino: {x_train.shape}, validação: {x_val.shape}, teste: {x_test.shape}")
    print(f"Memória das imagens: {(x_train.nbytes + x_val.nbytes + x_test.nbytes) / 1e6:.0f} MB")
    for split, y in [("Treino", y_train), ("Validação", y_val), ("Teste", y_test)]:
        print(f"{split}:", {name: int(np.sum(y == i)) for i, name in enumerate(CLASSES)})

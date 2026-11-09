"""Explicabilidade: mapas de onde a MobileNetV2 treinada no notebook 01 "olhou" para decidir (Grad-CAM).

A última camada convolucional da MobileNetV2 dá, para cada uma de 4×4 zonas da imagem (em imagens
128×128), 1280 números (features). O modelo faz a média de cada feature em todas as zonas e a camada final
dá um peso a cada feature para cada classe. O mapa de uma classe é, em cada zona, a soma das features
multiplicadas por esses pesos: as zonas com valores altos são as que mais fizeram subir a classe.

Em geral, o Grad-CAM calcula estes pesos com gradientes. Nesta rede (média das zonas seguida de uma só
camada final) os gradientes são exatamente os pesos da camada final divididos pelo nº de zonas, por isso
o mapa é o mesmo e não é preciso calcular gradientes (é o CAM original, de que o Grad-CAM é a generalização).
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .data import CLASSES


def load_model(path):
    return keras.models.load_model(path)


class GradCAM:
    """Calcula o mapa de uma imagem para uma classe, com a MobileNetV2 de build_mobilenet."""

    def __init__(self, model):
        self.model = model
        pooling = next(l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))
        self.features = keras.Model(model.inputs, pooling.input)   # imagem -> 4×4×1280 features
        self.kernel, self.bias = (w.numpy() for w in model.layers[-1].weights)   # pesos da camada final

    def __call__(self, image, class_index=None):
        """Devolve (mapa em [0, 1] com o tamanho da imagem, probabilidades). Sem class_index, explica a classe prevista."""
        features = self.features(image[None, ..., None], training=False).numpy()[0]
        logits = features.mean(axis=(0, 1)) @ self.kernel + self.bias   # o mesmo que o modelo calcula
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if class_index is None:
            class_index = int(np.argmax(probs))
        cam = np.maximum(features @ self.kernel[:, class_index], 0)   # só as zonas que fazem subir a classe
        cam = cam / cam.max() if cam.max() > 0 else cam
        cam = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear").numpy()[..., 0]
        return cam, probs


def lesion_overlap(cam, mask):
    """Compara o mapa com a lesão desenhada pelos médicos.

    - energy: fração da "importância" total do mapa que cai dentro da lesão;
    - area: fração da imagem ocupada pela lesão (o que teria um mapa que não olha para nada em especial).
    """
    energy = float((cam * mask).sum() / cam.sum()) if cam.sum() > 0 else 0.0   # mapa vazio: não aponta para nada
    return {"energy": energy, "area": float(mask.mean())}


def plot_grad_cam(explainer, images, y_true, indices, title, masks=None, per_row=4):
    """Para cada imagem: a ecografia e, ao lado, o Grad-CAM da classe prevista (vermelho = mais importante).

    Com masks, o contorno da lesão desenhada pelos médicos aparece a branco nas duas imagens.
    """
    indices = list(indices)
    if not indices:
        print(f"{title}: nenhuma imagem")
        return
    rows = (len(indices) + per_row - 1) // per_row
    fig, axs = plt.subplots(rows, 2 * per_row, figsize=(2.3 * 2 * per_row, 2.6 * rows + 0.6), squeeze=False)
    fig.suptitle(title)
    for ax in axs.flat:
        ax.axis("off")
    for k, i in enumerate(indices):
        ax_img, ax_cam = axs[k // per_row, 2 * (k % per_row)], axs[k // per_row, 2 * (k % per_row) + 1]
        cam, probs = explainer(images[i])
        pred = int(np.argmax(probs))
        ax_img.imshow(images[i], cmap="gray")
        ax_img.set_title(f"real: {CLASSES[y_true[i]]}", fontsize=9)
        ax_cam.imshow(images[i], cmap="gray")
        ax_cam.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        if masks is not None and masks[i].any():
            for ax in (ax_img, ax_cam):
                ax.contour(masks[i], levels=[0.5], colors="white", linewidths=1)
        ax_cam.set_title(f"previsto: {CLASSES[pred]} ({probs[pred]:.0%})", fontsize=9,
                         color="green" if pred == y_true[i] else "red")
    plt.tight_layout()
    plt.show()

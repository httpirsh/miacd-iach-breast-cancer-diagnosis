"""Explicabilidade: Grad-CAM sobre a MobileNetV2 treinada no notebook 01.

O Grad-CAM mostra que zonas da imagem mais pesaram na decisão do modelo:
1. a imagem passa pela rede até uma camada convolucional, que dá um "mapa" de features (ex.: 4×4×1280);
2. calcula-se o gradiente da probabilidade da classe em relação a esse mapa: quanto cada feature, em cada
   zona, faz subir essa probabilidade;
3. cada feature recebe um peso (a média do seu gradiente) e somam-se os mapas pesados;
4. ficam só os valores positivos (zonas que aumentam a probabilidade da classe), normalizados para [0, 1].
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .data import CLASSES

# Última camada convolucional da MobileNetV2 (mapa 4×4 em imagens 128×128). Camadas anteriores dão mapas
# mais finos (8×8), mas acendem os cantos da imagem (efeito do padding) e coincidem menos com a lesão.
LAYER = "out_relu"


def load_model(path):
    return keras.models.load_model(path)


class GradCAM:
    """Calcula o mapa Grad-CAM de uma imagem para uma classe, com a MobileNetV2 de build_mobilenet."""

    def __init__(self, model, layer=LAYER):
        self.model = model
        index = next(i for i, l in enumerate(model.layers) if isinstance(l, keras.Model))
        self.preprocess = model.layers[1:index]   # repetir o canal cinzento 3 vezes e reescalar para [-1, 1]
        base = model.layers[index]
        self.head = model.layers[index + 1:]      # pooling, dropout e camada de saída
        # A rede base devolve o mapa da camada escolhida e a sua saída final
        self.base = keras.Model(base.inputs, [base.get_layer(layer).output, base.output])

    def predict(self, image):
        """Probabilidades das 3 classes para uma imagem (size × size, valores em [0, 1])."""
        return self.model.predict(image[None, ..., None], verbose=0)[0]

    def __call__(self, image, class_index=None):
        """Devolve (mapa em [0, 1] com o tamanho da imagem, probabilidades). Sem class_index, explica a classe prevista."""
        x = tf.convert_to_tensor(image[None, ..., None], dtype=tf.float32)
        for layer in self.preprocess:
            x = layer([x, x, x]) if isinstance(layer, keras.layers.Concatenate) else layer(x)
        with tf.GradientTape() as tape:
            feature_map, out = self.base(x, training=False)
            tape.watch(feature_map)
            for layer in self.head[:-1]:
                out = layer(out, training=False)
            dense = self.head[-1]
            logits = tf.matmul(out, dense.kernel) + dense.bias   # score antes do softmax
            probs = tf.nn.softmax(logits)[0]
            if class_index is None:
                class_index = int(tf.argmax(probs))
            # Usa-se o score antes do softmax: quando o modelo tem ~100% de certeza, o softmax satura e os
            # gradientes da probabilidade ficam quase nulos.
            score = logits[0, class_index]
        grads = tape.gradient(score, feature_map)[0]               # gradiente em cada zona e feature
        weights = tf.reduce_mean(grads, axis=(0, 1))               # importância de cada feature
        cam = tf.nn.relu(tf.reduce_sum(feature_map[0] * weights, axis=-1)).numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        size = image.shape[:2]
        cam = tf.image.resize(cam[..., None], size, method="bilinear").numpy()[..., 0]
        return cam, probs.numpy()


def lesion_overlap(cam, mask):
    """Compara o Grad-CAM com a lesão desenhada pelos médicos.

    - energy: fração da "importância" total do mapa que cai dentro da lesão;
    - area: fração da imagem ocupada pela lesão (o que um mapa uniforme, que não olha para nada, teria);
    - hit: o ponto mais importante do mapa está dentro da lesão? ("pointing game")
    """
    energy = float((cam * mask).sum() / cam.sum()) if cam.sum() > 0 else 0.0   # mapa vazio: não aponta para nada
    return {"energy": energy, "area": float(mask.mean()),
            "hit": bool(mask.flat[np.argmax(cam)])}


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

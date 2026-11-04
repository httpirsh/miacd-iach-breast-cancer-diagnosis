"""Modelos: a CNN treinada de raiz e a MobileNetV2 pré-treinada (transfer learning)."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def compile_model(model, learning_rate):
    # jit_compile=False: sem isto o TensorFlow recompila o modelo (XLA) sempre que o nº de imagens muda,
    # o que é lento e vai gastando RAM até o processo ser morto.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="categorical_crossentropy",
                  metrics=["accuracy"], jit_compile=False)
    return model


def build_cnn(size=128, n_classes=3, learning_rate=0.001):
    """CNN treinada de raiz: a arquitetura do grupo de 2024, com duas correções.

    - learning rate 0.001 em vez de 0.00001 (com 0.00001 o modelo quase não aprendia em 15 épocas);
    - ReLU em vez de sigmoid na camada Dense escondida (a sigmoid satura e atrasa a aprendizagem).
    """
    model = keras.Sequential([
        layers.Input((size, size, 1)),
        layers.Conv2D(32, (4, 4), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (4, 4), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (4, 4), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),

        layers.Conv2D(32, (4, 4), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(n_classes, activation="softmax"),
    ])
    return compile_model(model, learning_rate)


def _mobilenet_input(size):
    """Entrada em escala de cinzento adaptada ao que a MobileNetV2 espera: 3 canais com valores em [-1, 1]."""
    inputs = layers.Input((size, size, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])   # repetir o canal cinzento 3 vezes (RGB)
    x = layers.Rescaling(2.0, offset=-1.0)(x)              # [0, 1] -> [-1, 1]
    return inputs, x


def build_mobilenet(size=128, n_classes=3, learning_rate=0.001):
    """MobileNetV2 pré-treinada no ImageNet, com uma nova camada de saída para as nossas 3 classes.

    Começa com a MobileNetV2 congelada (só a nova camada aprende); depois pode afinar-se o topo com
    unfreeze_top().
    """
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs, x = _mobilenet_input(size)
    x = base(x, training=False)    # training=False: as BatchNormalization da rede base não mudam
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return compile_model(keras.Model(inputs, outputs), learning_rate)


def unfreeze_top(model, n_layers=30, learning_rate=0.0001):
    """Fine-tuning: deixa treinar as últimas n_layers da MobileNetV2, com um learning rate 10x menor.

    As BatchNormalization ficam congeladas, como recomenda a documentação do Keras para fine-tuning.
    """
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return compile_model(model, learning_rate)


def fit(model, x_train, y_train, x_val, y_val, max_epochs=60, patience=10, batch_size=32):
    """Treina com early stopping: para quando a loss de validação não melhora durante `patience` épocas
    e volta aos pesos da melhor época. Só a validação decide quando parar; o teste não é usado."""
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, start_from_epoch=5,
                                                   restore_best_weights=True)
    return model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs, batch_size=batch_size,
                     callbacks=[early_stopping], verbose=0)


def extract_features(*image_sets, batch_size=32):
    """Passa as imagens pela MobileNetV2 congelada e devolve um vetor de 1280 números por imagem.

    Estes vetores (features) resumem o conteúdo da imagem. Usados no Active Learning, onde o
    classificador é treinado de novo a cada iteração e treinar a rede inteira seria demasiado lento.
    Aceita vários conjuntos (ex.: treino, validação, teste) e devolve uma lista com as features de cada um.
    """
    size = image_sets[0].shape[1]
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights="imagenet",
                                          pooling="avg")
    features = []
    for images in image_sets:
        rgb = np.repeat(images[..., None], 3, axis=-1) * 2.0 - 1.0   # cinzento -> 3 canais em [-1, 1]
        features.append(base.predict(rgb, batch_size=batch_size, verbose=0))
    del base
    keras.backend.clear_session()
    return features

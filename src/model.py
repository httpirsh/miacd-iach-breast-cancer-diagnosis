"""Modelo de CNN usado tanto no Passive Learning como no Active Learning."""

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_cnn(size=224, n_classes=3, learning_rate=0.00001):
    model = Sequential([
        Input((size, size, 1)),
        Conv2D(32, (4, 4), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (4, 4), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (4, 4), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(32, (4, 4), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(32, activation="sigmoid"),

        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

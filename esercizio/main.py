import numpy as np
from tensorflow.keras.models import Sequential

X_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
classi = np.load("data/class.npy")




"""
class Example:
    def __init__(self) -> None:
        self.model = ...  # Modello con Sequential o con Model di tensorflow
        self.class_array = np.load("data/class.npy")

    def build(self):
        self.model.build()

    def compile(self):
        self.model.compile(
            optimizer=...,  # Ottimizzatore
            loss=...,  # Loss per la classificazione
        )

    def fit(self, x_train, y_train):
        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=...,  # Numero di epoch
            batch_size=...,  # Dimensione batch
        )

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, x):
        prediction = self.model.predict(x, verbose=0)[0]
        top = np.argsort(prediction)[-5:][::-1]
        top_encoded = self.class_array[top]
        return prediction, top_encoded
"""
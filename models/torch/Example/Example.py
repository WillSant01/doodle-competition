import torch
import numpy as np

import torch.nn as nn
import pytorch_lightning as pl


class AlexNet(pl.LightningModule):
    def __init__(
        self,
    ):
        super(AlexNet, self).__init__()
        self.model = ...  # Modello con Sequential o con Model di pytorch

        self.class_array = np.load("data/class.npy")

        self.loss = ...  # Loss per la classificazione di pytorch

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(dtype=torch.float32)
        x = torch.reshape(x, (-1, 1, 28, 28))
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self.forward(X)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        out = self.forward(X)
        loss = self.loss(out, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(
        self,
    ):
        opt = torch.optim.AdamW(params=self.parameters(), lr=...)  # Ottimizzatore

        return {"optimizer": opt}

    def predict(self, x):
        with torch.no_grad():
            prediction = self.forward(x).detach().numpy()[0]
        top = np.argsort(prediction)[-5:][::-1]
        top_encoded = self.class_array[top]
        return prediction, top_encoded

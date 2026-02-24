import numpy as np
import time
from collections import deque

from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================
# Residual Block (MLP-style)
# ======================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + identity
        return self.relu(out)


# ======================================================
# ResNet-like classifier (dla danych wektorowych)
# ======================================================
class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, 128)

        self.blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.blocks(x)
        return self.fc_out(x)


# ======================================================
# Sliding Window ResNet (uniwersalny)
# ======================================================
class SlidingWindowPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, window_size=5, lr=1e-3, epochs=10):
        self.window_size = window_size
        self.lr = lr
        self.epochs = epochs

        self.classes_ = np.arange(10)
        self._is_initialized = False

        self.train_times_ = []
        self.memory_usage_ = []

    # --------------------------------------------------
    # Uniwersalne przygotowanie danych
    # --------------------------------------------------
    def _prepare_X(self, X):
        X = np.asarray(X)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X

    # --------------------------------------------------
    # Inicjalizacja / reset modelu
    # --------------------------------------------------
    def _init_model(self, input_dim):
        self.model = ResNetClassifier(
            input_dim=input_dim,
            num_classes=len(self.classes_)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self._is_initialized = True
        self.input_dim_ = input_dim

    # --------------------------------------------------
    # Trening jednego chunka
    # --------------------------------------------------
    def _train(self, X, y):
        X = self._prepare_X(X)

        # jeśli zmienił się wymiar cech → reset
        if self._is_initialized and X.shape[1] != self.input_dim_:
            print("RESET")
            self._init_model(X.shape[1])

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

    # --------------------------------------------------
    # partial_fit (sliding window)
    # --------------------------------------------------
    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, "buffer_"):
            self.buffer_ = deque(maxlen=self.window_size)
            self.k_ = 0
            if classes is not None:
                self.classes_ = classes

        t_start = time.perf_counter()

        self.buffer_.append((X, y))

        mem = sum(c[0].nbytes + c[1].nbytes for c in self.buffer_)
        self.memory_usage_.append(mem)

        Xp = self._prepare_X(X)

        if not self._is_initialized:
            self._init_model(Xp.shape[1])

        # ==================================================
        # k < L → inkrementalnie
        # ==================================================
        if self.k_ < self.window_size:
            self._train(X, y)

        # ==================================================
        # k ≥ L → RESET + trening od zera na oknie
        # ==================================================
        else:
            Xw = np.vstack([self._prepare_X(c[0]) for c in self.buffer_])
            yw = np.hstack([c[1] for c in self.buffer_])

            self._init_model(Xw.shape[1])
            self._train(Xw, yw)

        t_end = time.perf_counter()
        self.train_times_.append(t_end - t_start)

        self.k_ += 1
        return self

    # --------------------------------------------------
    # Predykcja
    # --------------------------------------------------
    def predict(self, X):
        if not self._is_initialized:
            return np.random.choice(self.classes_, size=len(X))

        X = self._prepare_X(X)

        # zabezpieczenie
        if X.shape[1] != self.input_dim_:
            raise ValueError(
                f"Niezgodny wymiar cech: {X.shape[1]} ≠ {self.input_dim_}"
            )

        X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)

        return logits.argmax(dim=1).cpu().numpy()
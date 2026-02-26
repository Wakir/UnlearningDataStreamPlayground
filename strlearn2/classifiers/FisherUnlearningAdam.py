import numpy as np
import time
from collections import deque

from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================
# Residual Block
# ======================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc2(self.relu(self.fc1(x))) + x)


# ======================================================
# ResNet-like classifier
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
        return self.fc_out(self.blocks(self.fc_in(x)))


# ======================================================
# Empirical Fisher Unlearning (Adam-compatible)
# ======================================================
class FisherUnlearningAdam(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        window_size=5,
        lr=1e-3,
        unlearning_rate=1.0,
        fisher_eps=1e-6,
        epochs=1
    ):
        self.window_size = window_size
        self.lr = lr
        self.unlearning_rate = unlearning_rate*lr
        self.fisher_eps = fisher_eps
        self.epochs = epochs

        self.classes_ = np.arange(10)
        self._is_initialized = False

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.train_times_ = []
        self.memory_usage_ = []

    # --------------------------------------------------
    def _prepare_X(self, X):
        X = np.asarray(X)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X

    # --------------------------------------------------
    def _init_model(self, input_dim):
        self.model = ResNetClassifier(
            input_dim=input_dim,
            num_classes=len(self.classes_)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        self._is_initialized = True
        self.input_dim_ = input_dim

    # --------------------------------------------------
    # Compute gradients + empirical Fisher
    # --------------------------------------------------
    def _compute_grad_and_fisher(self, X, y):
        X = torch.from_numpy(self._prepare_X(X)).float().to(self.device)
        y = torch.from_numpy(y).long().to(self.device)

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(X), y)
        loss.backward()

        grads = []
        fisher = []

        for p in self.model.parameters():
            g = p.grad.detach().clone()
            grads.append(g)
            fisher.append(g * g)  # diagonal empirical Fisher

        return grads, fisher

    # --------------------------------------------------
    # Normal Adam training
    # --------------------------------------------------
    def _train_chunk(self, X, y):
        grads, fisher = self._compute_grad_and_fisher(X, y)

        self.optimizer.step()

        return grads, fisher

    # --------------------------------------------------
    # Fisher-based unlearning
    # --------------------------------------------------
    def _unlearn(self, grads, fisher):
        with torch.no_grad():
            for p, g, f in zip(self.model.parameters(), grads, fisher):
                p.add_(
                    self.unlearning_rate * g / (torch.sqrt(f) + self.fisher_eps)
                )

    # --------------------------------------------------
    # partial_fit
    # --------------------------------------------------
    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, "buffer_"):
            self.buffer_ = deque(maxlen=self.window_size)
            self.k_ = 0
            if classes is not None:
                self.classes_ = classes

        t0 = time.perf_counter()

        Xp = self._prepare_X(X)
        if not self._is_initialized:
            self._init_model(Xp.shape[1])

        if self.k_ < self.window_size:
            gf = self._train_chunk(X, y)
            self.buffer_.append(gf)
        else:
            old_grads, old_fisher = self.buffer_.popleft()
            self._unlearn(old_grads, old_fisher)

            gf = self._train_chunk(X, y)
            self.buffer_.append(gf)

        self.memory_usage_.append(
            sum(
                sum(g.numel() * g.element_size() for g in gf[0])
                for gf in self.buffer_
            )
        )

        self.train_times_.append(time.perf_counter() - t0)
        self.k_ += 1
        return self

    # --------------------------------------------------
    def predict(self, X):
        if not self._is_initialized:
            return np.random.choice(self.classes_, size=len(X))

        X = self._prepare_X(X)
        X = torch.from_numpy(X).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)

        return logits.argmax(dim=1).cpu().numpy()


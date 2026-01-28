import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from strlearn.evaluators import TestThenTrain
from sklearn.datasets import fetch_openml
import time

class FisherUnlearningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, window_size=5, fisher_decay=0.9, eps=1e-6):
        self.window_size = window_size
        self.fisher_decay = fisher_decay
        self.eps = eps

        self.classes_ = np.arange(10)
        self.is_fitted = False
        self.k = 0

        self.window_deltas_ = []    # Δθ per chunk
        self.fisher_ = None         # diagonal Fisher

        self.train_times_ = []
        self.memory_usage_ = []

        self._init_model()

    def _init_model(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1,
            solver="sgd",
            learning_rate_init=0.01,
            momentum=0.0,      # WAŻNE
            random_state=42
        )
        self.is_fitted = False

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _get_weights(self):
        return [w.copy() for w in self.model.coefs_] + \
               [b.copy() for b in self.model.intercepts_]

    def _set_weights(self, weights):
        n = len(self.model.coefs_)
        self.model.coefs_ = weights[:n]
        self.model.intercepts_ = weights[n:]

    def _flatten(self, weights):
        return np.concatenate([w.ravel() for w in weights])

    def _unflatten(self, flat, template):
        out = []
        idx = 0
        for w in template:
            size = w.size
            out.append(flat[idx:idx+size].reshape(w.shape))
            idx += size
        return out

    # --------------------------------------------------
    # API
    # --------------------------------------------------
    def partial_fit(self, X, y, classes=None):
        t0 = time.perf_counter()

        # --- zapamiętaj wagi przed ---
        if self.is_fitted:
            theta_before = self._flatten(self._get_weights())

        # --- train ---
        self.model.partial_fit(X, y, classes=self.classes_)
        self.is_fitted = True

        theta_after = self._flatten(self._get_weights())
        delta = theta_after - theta_before if self.k > 0 else np.zeros_like(theta_after)

        # --- update Fisher ---
        if self.fisher_ is None:
            self.fisher_ = delta**2
        else:
            self.fisher_ = (
                self.fisher_decay * self.fisher_
                + (1 - self.fisher_decay) * delta**2
            )

        self.window_deltas_.append(delta)

        # --- UNLEARNING ---
        if len(self.window_deltas_) > self.window_size:
            old_delta = self.window_deltas_.pop(0)
            theta_new = theta_after - old_delta / (self.fisher_ + self.eps)
            self._set_weights(self._unflatten(theta_new, self._get_weights()))

        # --- logging ---
        self.k += 1
        t1 = time.perf_counter()

        self.train_times_.append(t1 - t0)
        mem = sum(X.nbytes for X in self.window_deltas_)
        self.memory_usage_.append(mem)

        return self

    def predict(self, X):
        if not self.is_fitted:
            return np.random.choice(self.classes_, size=len(X))
        return self.model.predict(X)


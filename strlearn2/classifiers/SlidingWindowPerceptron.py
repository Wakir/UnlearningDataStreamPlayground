import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time
from collections import deque


class SlidingWindowPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.classes_ = np.arange(10)
        self._is_initialized = False

        self.window_X = []
        self.window_y = []
        self.k = 0
        self.train_times_ = []      # tylko uczenie
        self.memory_usage_ = []

        self._init_model()

    def _init_model(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1,
            solver="adam",
            warm_start=False,
            random_state=42
        )
        self._is_initialized = False

    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, "buffer_"):
            self.buffer_ = deque(maxlen=self.window_size)
            self.k_ = 0
            self.classes_ = classes
            self._init_model()
            self.train_times_ = []      # tylko uczenie
            self.memory_usage_ = []

        t_train_start = time.perf_counter()

        # --- zapamiętaj chunk ---
        self.buffer_.append((X, y))
        mem = sum(
            c[0].nbytes + c[1].nbytes
            for c in self.buffer_
        )
        self.memory_usage_.append(mem)


        # ==================================================
        # k < L  → inkrementalne uczenie
        # ==================================================
        if self.k_ < self.window_size:
            if not self._is_initialized:
                self.model.partial_fit(X, y, classes=self.classes_)
                self._is_initialized = True
            else:
                self.model.partial_fit(X, y)

        # ==================================================
        # k ≥ L → RESET + trening od zera na oknie
        # ==================================================
        else:
            self._init_model()

            Xw = np.vstack([c[0] for c in self.buffer_])
            yw = np.hstack([c[1] for c in self.buffer_])

            print(len(yw))

            self.model.partial_fit(Xw, yw, classes=self.classes_)
            self._is_initialized = True

        t_train_end = time.perf_counter()
        self.train_times_.append(t_train_end - t_train_start)
        self.k_ += 1
        return self

    def predict(self, X):
        if not self._is_initialized:
            # losowe predykcje
            return np.random.choice(self.classes_, size=len(X))
        return self.model.predict(X)

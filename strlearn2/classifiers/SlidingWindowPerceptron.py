import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time


class SlidingWindowPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.classes_ = np.arange(10)
        self.is_fitted = False

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
            random_state=42
        )
        self.is_fitted = False

    def partial_fit(self, X, y, classes=None):
        t_train_start = time.perf_counter()
        self.window_X.append(X)
        self.window_y.append(y)

        if len(self.window_X) > self.window_size:
            self.window_X.pop(0)
            self.window_y.pop(0)

        if self.k < self.window_size:
            self.model.partial_fit(X, y, classes=self.classes_)
        else:
            self._init_model()
            Xw = np.vstack(self.window_X)
            yw = np.hstack(self.window_y)
            self.model.partial_fit(Xw, yw, classes=self.classes_)

        self.is_fitted = True
        self.k += 1
        t_train_end = time.perf_counter()
        mem_X = sum(X.nbytes for X in self.window_X)
        mem_y = sum(y.nbytes for y in self.window_y)
        mem = mem_X + mem_y
        self.memory_usage_.append(mem)
        self.train_times_.append(t_train_end - t_train_start)
        return self

    def predict(self, X):
        if not self.is_fitted:
            # losowe predykcje
            return np.random.choice(self.classes_, size=len(X))
        return self.model.predict(X)

import numpy as np
import time
from collections import deque
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class FisherUnlearningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            window_size=5,
            unlearning_rate=0.1,
            eps=1e-6
        ):
            self.window_size = window_size
            self.unlearning_rate = unlearning_rate
            self.eps = eps

            self.classes_ = np.arange(10)
            self.buffer_ = deque(maxlen=window_size)

            self.theta_prev_ = None
            self.k_ = 0
            self.is_fitted = False

            self.train_times_ = []
            self.memory_usage_ = []

            self._init_model()

    # ==================================================
    # Model
    # ==================================================
    def _init_model(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(100,),
            solver="adam",
            max_iter=1,
            warm_start=False,
            random_state=42
        )
        self.is_fitted = False

    # ==================================================
    # Utils
    # ==================================================
    def _get_last_layer(self):
        W = self.model.coefs_[-1]
        b = self.model.intercepts_[-1]
        return W.copy(), b.copy()

    def _set_last_layer(self, W, b):
        self.model.coefs_[-1] = W
        self.model.intercepts_[-1] = b

    # ==================================================
    # Partial fit
    # ==================================================
    def partial_fit(self, X, y, classes=None):
        t0 = time.perf_counter()
        if not hasattr(self, "classes_"):
            if classes is None:
                raise ValueError("classes must be provided on first call")
            self.classes_ = np.array(classes)
        if self.k_ < self.window_size:
            # --- TRAIN ---
            if not self.is_fitted:
                self.model.partial_fit(X, y, classes=self.classes_)
                self.is_fitted = True
            else:
                self.model.partial_fit(X, y, classes=self.classes_)

        # --- DELTA last layer ---
        W, b = self._get_last_layer()

        if self.theta_prev_ is not None:
            W_prev, b_prev = self.theta_prev_
            delta_W = W - W_prev
            delta_b = b - b_prev
        else:
            delta_W = np.zeros_like(W)
            delta_b = np.zeros_like(b)

        # --- UNLEARNING (opóźniony start!) ---
        if self.k_ >= self.window_size:
            delta_old_W, delta_old_b = self.buffer_.popleft()

            alpha = self.unlearning_rate
            W -= alpha * delta_old_W
            b -= alpha * delta_old_b

            self._set_last_layer(W, b)
            self.model.partial_fit(X, y, classes=self.classes_)

        # --- BUFFER ---
        self.buffer_.append((delta_W.copy(), delta_b.copy()))

        self.theta_prev_ = (W.copy(), b.copy())
        self.k_ += 1

        # --- LOGGING ---
        self.train_times_.append(time.perf_counter() - t0)
        self.memory_usage_.append(
            sum(dW.nbytes + db.nbytes for dW, db in self.buffer_)
        )

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)



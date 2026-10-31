import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from nanofit.metrics import mse
from nanofit.utils import add_dummy_feature

class MBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, t1=5, t2=50, batch_size=10, n_epochs=1000, random_state=None):
        self.t1 = t1
        self.t2 = t2
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs

    def _learning_schedule(self, t):
        return self.t1 / (t + self.t2)

    def fit(self, X, y):
        X = X.copy()
        X_b = add_dummy_feature(X)
        y = np.ravel(y).reshape(-1, 1)

        m = len(X_b)
        self.theta_ = np.ones((X_b.shape[1], 1))
        rng = np.random.default_rng(self.random_state)

        t = 0
        for epoch in range(self.n_epochs):
            order = rng.permutation(m)
            for i in range(0, m, self.batch_size):
                idx = order[i: i + self.batch_size]
                X_batch, y_batch = X_b[idx], y[idx]
                gradients = (2 / len(idx)) * X_batch.T @ (X_batch @ self.theta_ - y_batch) 
                eta = self._learning_schedule(t)
                self.theta_ = self.theta_ - eta * gradients
                t += 1

        flat_theta = self.theta_.ravel()
        self.intercept_ = flat_theta[0]
        self.coeff_ = flat_theta[1:]
        return self

    def predict(self, X):
        X_b = add_dummy_feature(X)
        return (X_b @ self.theta_).ravel()

    def score(self, X, y):
        y_preds = self.predict(X)
        return mse(y, y_preds)
        
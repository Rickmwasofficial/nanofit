import numpy as np
from nanofit.utils import add_dummy_feature
from nanofit.metrics import mse
from sklearn.base import BaseEstimator, RegressorMixin

class SGDRegressor(BaseEstimator, RegressorMixin):
    """
    Fit a linear model using Stochastic gradient descent as the optimization algorithm
    """
    def __init__(self, t1=5, t2=50, n_epochs=1000, random_state=None):
        self.t1 = t1
        self.t2 = t2
        self.n_epochs = n_epochs
        self.random_state = random_state

    def _learning_schedule(self, t):
        return self.t1 / (t + self.t2)

    def fit(self, X, y):
        X = X.copy()
        y = np.ravel(y).reshape(-1, 1)

        X_b = add_dummy_feature(X)
        m = len(X_b)
        self.theta_ = np.ones((X_b.shape[1], 1))
        rng = np.random.default_rng(self.random_state)

        t = 0
        for epoch in range(self.n_epochs):
            order = rng.permutation(m)
            for i in order:
                xi = X_b[i:i+1]
                yi = y[i:i+1]
                gradients = 2 * xi.T @ (xi @ self.theta_ - yi)
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
        
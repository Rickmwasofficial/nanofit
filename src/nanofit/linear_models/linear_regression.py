import numpy as np
from nanofit.utils import add_dummy_feature
from nanofit.metrics import mse
from sklearn.base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, method="normal", n_epochs = 1000, lr=0.01):
        """
        method: Training method for the model
        - normal -> using normal equation (PseudoInverse)
        - grad -> apply batch gradient descent optimization
        lr: size of steps for gradient descent
        n_epochs: Number of training loops for GD
        """
        self.method = method
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """
        X: Input Features
        y: Labels
        """
        X = X.copy()
        X_b = add_dummy_feature(X)
        y = np.ravel(y).reshape(-1, 1)

        if self.method == "normal":
            # use the formula theta = Xy where X is the pseudoinverse y is the labels
            X_ = np.linalg.pinv(X_b)
            self.theta_ = X_ @ y

        else:
            # use batch gradient descent
            m = len(X_b)
            self.theta_ = np.ones((X_b.shape[1], 1))
            for _ in range(self.n_epochs):
                gradients = (2 / m) * X_b.T @ (X_b @ self.theta_ - y)
                self.theta_ = self.theta_ - self.lr * gradients

        # force theta to be a 1D array so we do not guess it's shape
        flat_theta = self.theta_.ravel()
        self.intercept_ = flat_theta[0]
        self.coeff_ = flat_theta[1:]
        return self

    def predict(self, X):
        X_b = add_dummy_feature(X)
        return X_b @ self.theta_

    def score(self, X, y):
        y_preds = self.predict(X)
        return mse(y, y_preds)



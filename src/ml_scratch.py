   
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


class StandardScalerScratch:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps: float = 1e-8

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
                              
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScalerScratch not fitted.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LabelEncoderScratch:
    def __init__(self):
        self.classes_: Optional[List[str]] = None
        self.class_to_int: Dict[str, int] = {}
        self.unknown_value: int = -1

    def fit(self, values: List[str]):
        unique = []
        seen = set()
        for v in values:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        self.classes_ = unique
        self.class_to_int = {c: i for i, c in enumerate(unique)}
        return self

    def transform(self, values: List[str]) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LabelEncoderScratch not fitted.")
        return np.array([self.class_to_int.get(v, self.unknown_value) for v in values], dtype=int)

    def fit_transform(self, values: List[str]) -> np.ndarray:
        return self.fit(values).transform(values)

    def inverse_transform(self, ints: List[int]) -> List[str]:
        if self.classes_ is None:
            raise ValueError("LabelEncoderScratch not fitted.")
        out = []
        for i in ints:
            if 0 <= i < len(self.classes_):
                out.append(self.classes_[i])
            else:
                out.append(None)
        return out


def train_test_split_scratch(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    test_n = int(round(n * test_size))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


class LinearRegressionScratch:
    def __init__(self):
        self.coef_: Optional[np.ndarray] = None                             

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = add_bias(X)
                                               
        XtX = Xb.T @ Xb
        Xty = Xb.T @ y
        self.coef_ = np.linalg.pinv(XtX) @ Xty
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model not fitted.")
        Xb = add_bias(np.asarray(X, dtype=float))
        return Xb @ self.coef_


class RidgeRegressionScratch:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = add_bias(X)
        n_features = Xb.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0.0                          
        XtX = Xb.T @ Xb
        Xty = Xb.T @ y
        self.coef_ = np.linalg.pinv(XtX + self.alpha * I) @ Xty
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model not fitted.")
        Xb = add_bias(np.asarray(X, dtype=float))
        return Xb @ self.coef_


def soft_thresholding(w: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)


class LassoRegressionScratch:
    def __init__(self, alpha: float = 1.0, lr: float = 0.001, n_iter: int = 1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.w_: Optional[np.ndarray] = None                            

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Xb = add_bias(X)
        self.w_ = np.zeros(d + 1)
        for _ in range(self.n_iter):
            pred = Xb @ self.w_
            grad = (Xb.T @ (pred - y)) / n
                                              
            w_no_bias = self.w_[1:]
            w_no_bias = soft_thresholding(w_no_bias - self.lr * grad[1:], self.alpha * self.lr)
            self.w_[0] -= self.lr * grad[0]
            self.w_[1:] = w_no_bias
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise ValueError("Model not fitted.")
        Xb = add_bias(np.asarray(X, dtype=float))
        return Xb @ self.w_


         

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


           

def k_fold_split(n_samples: int, k: int, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


def cross_val_score_scratch(model_ctor, X: np.ndarray, y: np.ndarray, cv: int = 5, metric: str = 'r2', random_state: int = 42, **model_kwargs):
    scores = []
    for train_idx, test_idx in k_fold_split(X.shape[0], cv, random_state):
        model = model_ctor(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        if metric == 'r2':
            score = r2_score_scratch(y[test_idx], pred)
        elif metric == 'rmse':
            score = -rmse(y[test_idx], pred)                            
        else:       
            score = -mse(y[test_idx], pred)
        scores.append(score)
    return float(np.mean(scores)), float(np.std(scores))

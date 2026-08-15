#!/usr/bin/env python3
# Poisson generalized linear model (log link) fitted by Iteratively Reweighted
# Least Squares (IRLS) -- the Fisher-scoring algorithm behind every GLM solver.
# From scratch with numpy.
import numpy as np


def fit_irls(X, y, iters=50, tol=1e-10):
    # X: (n, d) features (no intercept). Returns beta with intercept first,
    # plus the log-likelihood at each iteration.
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xb = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xb.shape[1]); loglik = []
    for _ in range(iters):
        eta = Xb @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta + (y - mu) / mu                          # working response
        WX = Xb * mu[:, None]                            # weights = mu (Poisson)
        beta_new = np.linalg.solve(Xb.T @ WX, Xb.T @ (mu * z))
        loglik.append(float(np.sum(y * eta - mu)))       # up to a constant
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    return beta, loglik


def predict(X, beta):
    Xb = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    return np.exp(Xb @ beta)


def deviance_residuals(y, mu):
    # Signed deviance residuals for a Poisson GLM.
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    d = 2 * (term - (y - mu))
    return np.sign(y - mu) * np.sqrt(np.abs(d))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2, 200)
    mu = np.exp(0.5 + 1.2 * x)
    y = rng.poisson(mu)
    beta, _ = fit_irls(x[:, None], y)
    print("beta (intercept, slope):", beta.round(3))

#!/usr/bin/env python3
# Self-contained demo: a Poisson GLM fitted by Iteratively Reweighted Least
# Squares from scratch, validated against scikit-learn. Poisson regression is
# the right tool for count outcomes -- read counts, mutation counts, cell counts
# -- where ordinary linear regression is simply the wrong model.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from poisson_glm import fit_irls, predict, deviance_residuals

RNG = np.random.default_rng(0)
TRUE = np.array([0.5, 1.2])          # intercept, slope on the log scale


def main():
    os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
    x = RNG.uniform(0, 2.5, 250)
    mu = np.exp(TRUE[0] + TRUE[1] * x)
    y = RNG.poisson(mu)

    beta, loglik = fit_irls(x[:, None], y)
    sk = PoissonRegressor(alpha=0.0, fit_intercept=True, max_iter=500).fit(x[:, None], y)
    sk_beta = np.r_[sk.intercept_, sk.coef_]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: counts vs covariate with the fitted mean.
    a = ax[0, 0]
    a.scatter(x, y, s=12, alpha=0.5, color="#4C72B0", label="counts")
    gx = np.linspace(0, 2.5, 100)
    a.plot(gx, predict(gx[:, None], beta), color="#C44E52", lw=2, label="fitted mean (Poisson GLM)")
    a.plot(gx, np.exp(TRUE[0] + TRUE[1] * gx), "k--", lw=1, label="true mean")
    a.set_xlabel("covariate x"); a.set_ylabel("count y")
    a.set_title("Poisson regression fits the mean count"); a.legend(fontsize=8)

    # Panel 2: IRLS convergence.
    a = ax[0, 1]
    a.plot(loglik, "o-", color="#4C72B0")
    a.set_xlabel("IRLS iteration"); a.set_ylabel("log-likelihood")
    a.set_title(f"IRLS converges in {len(loglik)} steps")

    # Panel 3: coefficient comparison, scratch vs scikit-learn.
    a = ax[1, 0]
    labels = ["intercept", "slope"]; xi = np.arange(2)
    a.bar(xi - 0.25, beta, 0.25, label="IRLS (scratch)", color="#4C72B0")
    a.bar(xi + 0.00, sk_beta, 0.25, label="scikit-learn", color="#DD8452")
    a.bar(xi + 0.25, TRUE, 0.25, label="true", color="#55A868")
    a.set_xticks(xi); a.set_xticklabels(labels)
    a.set_title(f"Coefficients agree (max diff {np.abs(beta-sk_beta).max():.4f})"); a.legend(fontsize=8)

    # Panel 4: deviance residuals vs fitted.
    a = ax[1, 1]
    mu_hat = predict(x[:, None], beta)
    dres = deviance_residuals(y, mu_hat)
    a.scatter(mu_hat, dres, s=12, alpha=0.5, color="#8172B3")
    a.axhline(0, color="k", ls="--")
    a.set_xlabel("fitted mean"); a.set_ylabel("deviance residual")
    a.set_title("Residuals: no trend = good fit")

    fig.suptitle("Poisson GLM via IRLS, validated against scikit-learn (synthetic data)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig("figures/demo.png", dpi=120)

    pd.DataFrame({"parameter": labels, "true": TRUE, "irls": beta, "sklearn": sk_beta}).to_csv(
        "results/summary.csv", index=False)
    print(f"IRLS beta:    {beta.round(3)}")
    print(f"sklearn beta: {sk_beta.round(3)}")
    print(f"max coefficient difference: {np.abs(beta-sk_beta).max():.5f}")
    print("Wrote figures/demo.png and results/summary.csv")


if __name__ == "__main__":
    main()

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns

    return mo, np, pd, sns


@app.cell
def _(sns):
    sns.set_theme()
    return


@app.cell
def _(np):
    np.random.seed(42)
    m = 100
    n = 1
    X = 2 * np.random.rand(m, n)
    y = 4 + 3 * X + np.random.randn(m, n)

    X.flatten(), y.flatten()
    return X, y


@app.cell
def _(X, pd, sns, y):
    df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})

    sns.regplot(
        data=df,
        x="X",
        y="y",
        scatter_kws={"alpha": 0.3, "color": "teal"},
        line_kws={"color": "blue"},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $y\hat\ = θ_0 + θ_1x_1 + θ_2x_2 + ⋯ + θ_nx_n$

    $\theta\hat\ = (X^⊺X)^{-1} X^⊺y$
    """)
    return


@app.cell
def _(X, np, y):
    from sklearn.preprocessing import add_dummy_feature

    X_b = add_dummy_feature(X)

    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    theta_best
    return add_dummy_feature, theta_best


@app.cell
def _(add_dummy_feature, np, theta_best):
    X_new = np.array([[0], [2], [5]])
    X_new_b = add_dummy_feature(X_new)
    y_predict = X_new_b @ theta_best
    y_predict
    return


@app.cell
def _(np):
    norm = np.random.rand(4, 4)
    norm, np.linalg.pinv(norm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polynomial Regression
    """)
    return


@app.cell
def _(np):
    np.random.seed(42)
    return


@app.cell
def _(np):
    pr_m = 100
    pr_X = 6 * np.random.rand(pr_m, 1) - 3
    pr_y = 0.5 * pr_X**2 + pr_X + 2 + np.random.randn(pr_m, 1)

    pr_X, pr_y
    return pr_X, pr_y


@app.cell
def _(pd, pr_X, pr_y, sns):
    pr_df = pd.DataFrame({"X": pr_X.flatten(), "y": pr_y.flatten()})

    sns.relplot(data=pr_df, x="X", y="y", aspect=1.5)
    return


@app.cell
def _(pr_X):
    from sklearn.preprocessing import PolynomialFeatures

    poly_featrues = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_featrues.fit_transform(pr_X)
    pr_X, X_poly[0]
    return (X_poly,)


@app.cell
def _(X_poly, pr_y):
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, pr_y)
    lin_reg.intercept_, lin_reg.coef_
    return


if __name__ == "__main__":
    app.run()

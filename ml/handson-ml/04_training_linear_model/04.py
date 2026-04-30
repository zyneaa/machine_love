import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


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
    X = 2 * np.random.rand(m, 2)
    y = 4 + 3 * X + np.random.randn(m, 2)

    X, y
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
    X_b, X_b.T, theta_best
    return add_dummy_feature, theta_best


@app.cell
def _(add_dummy_feature, np, theta_best):
    X_new = np.array([[0, 2], [2, 2], [5, 4]])
    X_new_b = add_dummy_feature(X_new)
    y_predict =  X_new_b @ theta_best
    y_predict
    return


if __name__ == "__main__":
    app.run()

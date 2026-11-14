import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $w^Tx+b=w_1\ x_1+w_2x_2+......$
    """)
    return


@app.cell
def _(np):
    # dummy dataset dimensions
    np.random.seed(42)
    m = 100
    n = 3
    return m, n


@app.cell
def _(m, mo, n, np):
    w = np.random.randn(n)
    b = 0.0
    X = np.random.randn(m, n)

    decision_scores = X @ w + b
    y = (decision_scores >= 0).astype(int)

    mo.md(f"""
    **Dummy SVM inputs**

    | | shape |
    |---|---|
    | `X` (features) | {X.shape} |
    | `w` (weights) | {w.shape} |
    | `b` (bias) | {b} |

    First sample: `X[0] = {np.round(X[0], 3)}`
    """)
    return X, decision_scores, w, y


@app.cell
def _(X, decision_scores, w, y):
    w, X, decision_scores, y
    return


@app.cell
def _(w):
    (w.T @ w) / 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The following code has __w__ as weights, __s__ as summation of slack vars' vals, and C as hyperparameter
    """)
    return


@app.function
def minimize(w, s, C):
    return (w.T @ w) / 2 + (C * s)


if __name__ == "__main__":
    app.run()

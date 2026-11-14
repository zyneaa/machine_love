import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np


@app.cell
def _():
    n = 4
    return (n,)


@app.cell
def _(n, np):
    matrix = np.random.randint(1, 100, size=(n, n))
    matrix
    return (matrix,)


@app.cell
def _(matrix, np):
    det = np.linalg.det(matrix)
    det
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using Laplace Expansion
    """)
    return


@app.cell
def _(np):
    def minor(A, k, j):
        M = np.delete(np.delete(A, k, axis=0), j, axis=1)
        return np.linalg.det(M)

    return (minor,)


@app.cell
def _(matrix, minor, n):
    d = 0

    # Expansion along column j
    j = 0
    for k in range(n):
        d += (-1) ** (k + j) * matrix[k][j] * minor(matrix, k, j)

    d
    return


if __name__ == "__main__":
    app.run()

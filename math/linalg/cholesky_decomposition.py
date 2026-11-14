import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    rng = np.random.default_rng()
    return mo, np, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The following cell results in infinite loop
    """)
    return


@app.cell
def _(np, positive_definite_matrix, rng):
    n = 4
    not_positive_definite_matrices = []
    found = False
    positive_definite_matirx = None

    while found != True:
        arr = rng.random(size=(n, n))

        if not np.allclose(arr, arr.T):
            not_positive_definite_matrices.append(arr)
            continue

        found = True
        positive_definite_matirx = arr

    positive_definite_matrix
    return (n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The standard apporach, first we make a symmetirc matrix
    """)
    return


@app.cell
def _(n, rng):
    A = rng.random(size=(n, n))
    B = (A + A.T) / 2
    B
    return (B,)


@app.cell
def _(B, np):
    np.allclose(B, B.T)
    return


@app.cell
def _(B, np):
    C = None
    try:
        C = np.linalg.cholesky(B)

    except np.linalg.LinAlgError:
        print("failed")

    C
    return


if __name__ == "__main__":
    app.run()

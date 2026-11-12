import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    import numpy as np

    return (np,)


@app.cell
def _(np):
    np.random.seed(42)
    m = 100
    n = 10
    X = 2 * np.random.rand(m, n)
    y = 4 + 3 * X + np.random.randn(m, 1)

    X, y.flatten()
    return


if __name__ == "__main__":
    app.run()

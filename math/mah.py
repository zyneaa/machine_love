import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(np):
    theta = np.array([2, 3, 4, 5]).reshape(4, 1)
    theta
    return (theta,)


@app.cell
def _(np):
    x = np.array([12, 34, 10, 1]).reshape(4, 1)
    x
    return (x,)


@app.cell
def _(theta, x):
    theta.T.dot(x)
    return


@app.cell
def _(theta, x):
    theta[1].dot(x[1])
    return


if __name__ == "__main__":
    app.run()

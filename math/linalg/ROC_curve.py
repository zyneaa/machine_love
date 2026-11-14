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
    import matplotlib as ms

    return


@app.cell
def _():
    tpr = 10 / (10 + 3)
    tpr
    return


@app.cell
def _():
    fpr = 3 / (3+12)
    fpr
    return


if __name__ == "__main__":
    app.run()

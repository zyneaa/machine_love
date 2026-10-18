import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return (np,)


@app.cell
def _(np):
    labels = np.array(["cat", "dog", "bird", "fish"])
    return (labels,)


@app.cell
def _(labels, np):
    random_labels_actual = np.random.choice(labels, size=1000)
    return (random_labels_actual,)


@app.cell
def _(random_labels_actual):
    random_labels_actual[:3]
    return


@app.cell
def _(labels, np):
    random_labels_predicted = np.random.choice(labels, size=1000)
    return (random_labels_predicted,)


@app.cell
def _(random_labels_predicted):
    random_labels_predicted[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $L(θ)\ =\ \frac{1}{N} \sum_{n=1}^{N}∏(y_n\ \neq f(x_n; θ))$
    """)
    return


@app.cell
def _(np, random_labels_actual, random_labels_predicted):
    empirical_risk = np.mean(random_labels_actual != random_labels_predicted)
    empirical_risk
    return


if __name__ == "__main__":
    app.run()

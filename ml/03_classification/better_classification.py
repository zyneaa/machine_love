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
    import pandas as pd
    import matplotlib as plt
    import seaborn as sns

    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", as_frame=False)
    return (mnist,)


@app.cell
def _(mnist):
    mnist.keys()
    return


@app.cell
def _(mnist):
    X, y = mnist.data, mnist.target
    X, y
    return X, y


@app.cell
def _(X):
    X.shape
    return


@app.cell
def _(mnist):
    print(mnist.DESCR)
    return


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test):
    len(X_test)
    return


@app.cell
def _(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier

    knn_clf = KNeighborsClassifier(weights="uniform")
    knn_clf.fit(X_train, y_train)
    return (knn_clf,)


@app.cell
def _(X_test, knn_clf, y_test):
    baseline_accuracy = knn_clf.score(X_test, y_test)
    baseline_accuracy
    return


if __name__ == "__main__":
    app.run()

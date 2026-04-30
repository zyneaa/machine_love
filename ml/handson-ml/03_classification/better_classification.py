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
def _(y_test):
    y_test[:20]
    for num in range(len(y_test[:100])):
        print(num, y_test[num])
    return


@app.cell
def _(X_test, y_test):
    some_digit = 99

    ok = X_test[some_digit]
    y_test[some_digit]
    return (ok,)


@app.cell
def _(knn_clf, ok):
    pred = knn_clf.predict([ok])
    pred
    return


@app.cell
def _(X_train, knn_clf, y_train):
    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
    return (y_train_pred,)


@app.cell
def _(X_test, knn_clf, y_test):
    baseline_accuracy = knn_clf.score(X_test, y_test)
    baseline_accuracy
    return


@app.cell
def _(y_train, y_train_pred):
    from sklearn.metrics import precision_score, recall_score

    precision_score(y_train, y_train_pred, average="macro")
    return (recall_score,)


@app.cell
def _(recall_score, y_train, y_train_pred):
    recall_score(y_train, y_train_pred, average="macro")
    return


@app.cell
def _(si, y_train, y_train_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_train, y_train_pred, si)
    cm
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    return (plt,)


@app.cell
def _(plt, y_train, y_train_pred):
    from sklearn.metrics import ConfusionMatrixDisplay

    plt.rc('font', size=9)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
    plt.show()
    return (ConfusionMatrixDisplay,)


@app.cell
def _(ConfusionMatrixDisplay, plt, y_train, y_train_pred):
    plt.rc('font', size=9)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
    plt.show()
    return


@app.cell
def _(ConfusionMatrixDisplay, plt, y_train, y_train_pred):
    sample_weight = (y_train_pred != y_train)
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                            sample_weight=sample_weight)
    plt.show()
    return (sample_weight,)


@app.cell
def _(ConfusionMatrixDisplay, plt, sample_weight, y_train, y_train_pred):
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                            sample_weight=sample_weight,
                                            normalize="true", values_format=".0%")
    plt.show()
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns

    from sklearn.datasets import load_iris
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return LinearSVC, StandardScaler, load_iris, make_pipeline, sns


@app.cell
def _(load_iris, sns):
    iris = load_iris(as_frame=True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]]
    y = iris.target == 2

    plot_df = X.assign(species=iris.target_names[iris.target])
    plot_df["is_virginica"] = y

    g = sns.relplot(
        data=plot_df,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        style="species",
        palette="deep",
        height=6,
        aspect=1.3,
        s=90,
        edgecolor="black",
        linewidth=0.5,
    )
    g.fig.suptitle("Iris: Petal Length vs Petal Width", y=1.03)
    g.set_axis_labels("Petal Length (cm)", "Petal Width (cm)")
    g
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialFeatures, make_moons


@app.cell
def _(make_moons):
    X_moon, y_moon = make_moons(n_samples=100_000, noise=0.15, random_state=42)
    X_moon, y_moon
    return X_moon, y_moon


@app.cell
def _(X_moon, sns, y_moon):
    import numpy as np
    import pandas as pd

    moon_df = pd.DataFrame(X_moon, columns=["x1", "x2"])
    moon_df["class"] = np.where(y_moon == 0, "moon A", "moon B")

    g_moon = sns.relplot(
        data=moon_df,
        x="x1",
        y="x2",
        hue="class",
        style="class",
        palette="deep",
        height=6,
        aspect=1.3,
        s=10,
        edgecolor="black",
        linewidth=0.5,
    )
    g_moon.fig.suptitle("Moons Dataset", y=1.03)
    g_moon.set_axis_labels("x1", "x2")
    g_moon
    return


@app.cell
def _(LinearSVC, PolynomialFeatures, StandardScaler, make_pipeline):
    clf = make_pipeline(
        PolynomialFeatures(degree=3),
        StandardScaler(),
        LinearSVC(C=10, max_iter=10000, random_state=42),
    )
    clf
    return (clf,)


@app.cell
def _(X_moon, clf, y_moon):
    clf.fit(X_moon, y_moon)
    return


if __name__ == "__main__":
    app.run()

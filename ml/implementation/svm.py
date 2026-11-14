import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np):
    n = 2
    m = 500
    rng = np.random.default_rng(seed=42)
    return m, n, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generating dummy data for implementation
    """)
    return


@app.cell
def _(m, n, rng):
    features = rng.random(size=(m, n))
    features, len(features), features.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F2AFO
    """)
    return


@app.cell
def _(plt, rng):
    data_centered_at_0 = rng.normal(loc=0, scale=5, size=1000)
    data_centered_at_20 = rng.normal(loc=10, scale=1, size=1000)

    # Print the actual sample means
    print(f"Mean when loc=0:  {data_centered_at_0.mean():.2f}")
    print(f"Mean when loc=20: {data_centered_at_20.mean():.2f}")

    # Plot both distributions as overlapping histograms
    plt.hist(data_centered_at_0, bins=30, alpha=0.6, label="loc=0 (Mean ≈ 0)")
    plt.hist(
        data_centered_at_20, bins=30, alpha=0.6, label="loc=20 (Mean ≈ 20)"
    )

    plt.title("Effect of 'loc' on a Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    return


@app.cell
def _(m, n, rng):
    X_positive = rng.normal(loc=[m, n], scale=100, size=(m, n))
    X_positive
    return (X_positive,)


@app.cell
def _(m, n, rng):
    X_negative = rng.normal(loc=[-m, -n], scale=200, size=(m, n))
    X_negative
    return (X_negative,)


@app.cell
def _():
    import pandas as pd
    import seaborn as sns

    return pd, sns


@app.cell(hide_code=True, expand_output=True)
def _(X_negative, X_positive, np, pd, plt, sns):
    from matplotlib.lines import Line2D

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="deep",
        rc={
            "figure.facecolor": "#fbfbf9",
            "axes.facecolor": "#fbfbf9",
            "savefig.facecolor": "#fbfbf9",
            "axes.edgecolor": "#d6d8d3",
            "axes.labelcolor": "#334155",
            "grid.color": "#e7e8e3",
            "text.color": "#1f2937",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
        },
    )

    COLORS = {"positive": "#2563eb", "negative": "#dc2626"}
    points = pd.concat(
        [
            pd.DataFrame(
                {
                    "x1": X_positive[:, 0],
                    "x2": X_positive[:, 1],
                    "class": "positive",
                }
            ),
            pd.DataFrame(
                {
                    "x1": X_negative[:, 0],
                    "x2": X_negative[:, 1],
                    "class": "negative",
                }
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    sns.scatterplot(
        data=points,
        x="x1",
        y="x2",
        hue="class",
        palette=COLORS,
        s=28,
        alpha=0.55,
        linewidth=0,
        ax=ax,
    )

    for name, pts in (("positive", X_positive), ("negative", X_negative)):
        ax.scatter(
            *pts.mean(axis=0),
            marker="X",
            s=220,
            color=COLORS[name],
            edgecolor="white",
            linewidth=1.6,
            zorder=5,
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(
        f"Two Gaussian classes  ·  {len(X_positive)} points each  ·  σ = 0.8",
        fontsize=13,
        pad=12,
    )

    every = np.vstack([X_positive, X_negative])
    pad = 0.08 * float(np.ptp(every, axis=0).max())
    ax.set_xlim(every[:, 0].min() - pad, every[:, 0].max() + pad)
    ax.set_ylim(every[:, 1].min() - pad, every[:, 1].max() + pad)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            marker="X",
            ls="",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor="#475569",
            label="class mean",
        )
    )
    ax.legend(
        handles=handles,
        title="class",
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        borderpad=0.8,
    )

    sns.despine(ax=ax)
    fig
    return


@app.cell
def _(X_negative, X_positive, np):
    X = np.vstack([X_positive, X_negative])
    X
    return (X,)


@app.cell
def _(m, np):
    t = np.concatenate([np.ones(m), -np.ones(m)])
    t
    return (t,)


@app.cell
def _(X, rng, t):
    indices = rng.permutation(len(X))
    X_ = X[indices]
    t_ = t[indices]
    X_, t_
    return X_, t_


@app.cell
def _(X_, t_):
    X_[t_ == 1]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

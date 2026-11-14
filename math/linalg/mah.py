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
    import matplotlib.pyplot as plt
    import seaborn as sns


    def plot_linear_transformation(A, v):
        """A: 2x2 matrix (linear transformation). v: column vector (2x1) plotted in transformed space."""
        A = np.asarray(A, dtype=float)
        v = np.asarray(v, dtype=float).reshape(2, 1)
        v_t = A @ v

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 8))

        # original grid (uniform gray)
        lim = max(abs(v_t).max(), 5) * 1.5
        grid = np.arange(-lim, lim + 1, 1)
        for g in grid:
            ax.axhline(g, color="lightgray", lw=0.8, zorder=0)
            ax.axvline(g, color="lightgray", lw=0.8, zorder=0)

        # original axes
        ax.axhline(0, color="black", lw=2, zorder=1)
        ax.axvline(0, color="black", lw=2, zorder=1)

        # transformed grid lines: images of x and y unit lines, distinct colors
        ts = np.linspace(-lim, lim, 50)
        for g in grid:
            x_line = A @ np.stack([g * np.ones_like(ts), ts])
            y_line = A @ np.stack([ts, g * np.ones_like(ts)])
            ax.plot(x_line[0], x_line[1], color="steelblue", lw=1, alpha=0.8, zorder=2)
            ax.plot(y_line[0], y_line[1], color="tomato", lw=1, alpha=0.8, zorder=2)

        # transformed axes (images of basis vectors)
        e1 = A @ np.array([[1], [0]])
        e2 = A @ np.array([[0], [1]])
        ax.quiver(0, 0, e1[0, 0], e1[1, 0], angles="xy", scale_units="xy", scale=1,
                  color="navy", width=0.008, zorder=3, label="transformed x-axis (Âe1)")
        ax.quiver(0, 0, e2[0, 0], e2[1, 0], angles="xy", scale_units="xy", scale=1,
                  color="darkred", width=0.008, zorder=3, label="transformed y-axis (Âe2)")

        # vector v and its image
        ax.quiver(0, 0, v[0, 0], v[1, 0], angles="xy", scale_units="xy", scale=1,
                  color="green", width=0.01, zorder=4, label="v")
        ax.quiver(0, 0, v_t[0, 0], v_t[1, 0], angles="xy", scale_units="xy", scale=1,
                  color="purple", width=0.01, zorder=4, label="Av")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_title("Linear Transformation")
        plt.show()

    return (plot_linear_transformation,)


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


@app.cell
def _(plot_linear_transformation):
    # Example: shear matrix applied to a column vector
    A = [[1, 1], [0, 1]]
    v = [[1], [2]]
    plot_linear_transformation(A, v)
    return


if __name__ == "__main__":
    app.run()

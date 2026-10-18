import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vector Visualization

    Use the sliders to build 2D and 3D vectors and see them as arrows from the origin.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D vectors
    """)
    return


@app.cell
def _(mo):
    ux = mo.ui.slider(-5, 5, value=2, label="uₓ")
    uy = mo.ui.slider(-5, 5, value=5, label="uᵧ")
    vx = mo.ui.slider(-5, 5, value=3, label="vₓ")
    vy = mo.ui.slider(-5, 5, value=1, label="vᵧ")
    mo.hstack([mo.vstack([ux, uy]), mo.vstack([vx, vy])])
    return ux, uy, vx, vy


@app.cell
def _(np, plt, ux, uy, vx, vy):
    def plot_vectors2d(*vectors, labels=None, colors=("r", "b")):
        fig, ax = plt.subplots()
        for i, vec in enumerate(vectors):
            color = colors[i % len(colors)]
            ax.quiver(
                0, 0, vec[0], vec[1],
                angles="xy", scale_units="xy", scale=1,
                color=color, zorder=5,
            )
            if labels:
                ax.text(vec[0] * 1.05, vec[1] * 1.05, labels[i], color=color, fontsize=16)
        lim = max(1, max(np.abs(v).max() for v in vectors)) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_aspect("equal")
        ax.grid(True)
        return fig

    u = np.array([ux.value, uy.value])
    v = np.array([vx.value, vy.value])
    plot_vectors2d(u, v, labels=["u", "v"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projection via vector addition

    The projection of **v** onto **u** splits **v** into two perpendicular parts whose sum is **v**:

    $\textbf{proj}_{\textbf{u}}{\textbf{v}} = \dfrac{\textbf{u} \cdot \textbf{v}}{\left \Vert \textbf{u} \right \| ^2} \times \textbf{u},
    \qquad
    \textbf{v} = \textbf{proj}_{\textbf{u}}{\textbf{v}} + \textbf{perp}$

    Move the sliders above: the black arrow is the projection, the green arrow is the leftover
    (perpendicular) part added onto the projection's tip to reconstruct **v**.
    """)
    return


@app.cell
def _(np, plt, ux, uy, vx, vy):
    _u = np.array([ux.value, uy.value], dtype=float)
    _v = np.array([vx.value, vy.value], dtype=float)

    fig, ax = plt.subplots()
    ax.quiver(0, 0, _u[0], _u[1], angles="xy", scale_units="xy", scale=1, color="r", zorder=5)
    ax.quiver(0, 0, _v[0], _v[1], angles="xy", scale_units="xy", scale=1, color="b", zorder=4)

    norm_u = np.linalg.norm(_u)
    if norm_u > 0:
        _u_hat = _u / norm_u
        _proj = _v.dot(_u_hat) * _u_hat
        _perp = _v - _proj

        ax.quiver(0, 0, _proj[0], _proj[1], angles="xy", scale_units="xy", scale=1,
                  color="k", linestyle=":", zorder=6)
        ax.quiver(_proj[0], _proj[1], _perp[0], _perp[1], angles="xy", scale_units="xy", scale=1,
                  color="g", zorder=6)
        ax.plot([_v[0], _proj[0]], [_v[1], _proj[1]], "b:", linewidth=1, zorder=3)

        ax.text(_proj[0] * 0.9, _proj[1] * 0.9, r"$proj_u v$", color="k", fontsize=14)
        ax.text(_proj[0] + _perp[0] * 0.5, _proj[1] + _perp[1] * 0.5, "perp", color="g", fontsize=14)

    ax.text(_u[0] * 1.05, _u[1] * 1.05, "u", color="r", fontsize=16)
    ax.text(_v[0] * 1.05, _v[1] * 1.05, "v", color="b", fontsize=16)

    _lim = max(1, np.abs(np.concatenate([_u, _v])).max()) * 1.2
    ax.set_xlim(-_lim, _lim)
    ax.set_ylim(-_lim, _lim)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D vectors
    """)
    return


@app.cell
def _(mo):
    ax_ = mo.ui.slider(-10, 10, value=1, label="aₓ")
    ay_ = mo.ui.slider(-10, 10, value=2, label="aᵧ")
    az_ = mo.ui.slider(-10, 10, value=8, label="a_z")
    bx_ = mo.ui.slider(-10, 10, value=5, label="bₓ")
    by_ = mo.ui.slider(-10, 10, value=6, label="bᵧ")
    bz_ = mo.ui.slider(-10, 10, value=3, label="b_z")
    mo.hstack(
        [mo.vstack([ax_, ay_, az_]), mo.vstack([bx_, by_, bz_])],
        justify="start",
    )
    return ax_, ay_, az_, bx_, by_, bz_


@app.cell
def _(ax_, ay_, az_, bx_, by_, bz_, np, plt):
    def plot_vectors3d(*vectors, labels=None, colors=("r", "b")):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, vec in enumerate(vectors):
            color = colors[i % len(colors)]
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, arrow_length_ratio=0.1)
            if labels:
                ax.text(vec[0], vec[1], vec[2], labels[i], color=color, fontsize=16)
        lim = max(1, max(np.abs(v).max() for v in vectors)) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return fig

    a = np.array([ax_.value, ay_.value, az_.value])
    b = np.array([bx_.value, by_.value, bz_.value])
    plot_vectors3d(a, b, labels=["a", "b"])
    return


if __name__ == "__main__":
    app.run()

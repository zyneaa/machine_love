import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(np):
    rng = np.random.default_rng(seed=42)
    return (rng,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Singular Value Decomposition

    Every matrix $A \in \mathbb{R}^{m \times n}$ factors as

    $$
    A = U \Sigma V^\top
    $$

    where

    - $U \in \mathbb{R}^{m \times m}$ has **orthonormal columns** — the *left* singular vectors,
    - $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$,
    - $V \in \mathbb{R}^{n \times n}$ has orthonormal columns — the *right* singular vectors.

    Unlike an eigendecomposition, this exists for **any** matrix, square or not, symmetric or not.
    Like an eigendecomposition, it has a clean geometric reading: $V^\top$ rotates, $\Sigma$ stretches,
    $U$ rotates back.
    """)
    return


@app.cell
def _(rng):
    A = rng.random(size=(3, 3))
    A
    return (A,)


@app.cell
def _(A, np):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U, S, Vt
    return S, U, Vt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The four pieces of the factorization

    `numpy` hands back the *economy* SVD: `S` is a length-$\min(m, n)$ vector of singular values
    rather than a full diagonal matrix, so rebuild $\Sigma$ with `np.diag(S)` when you need it.

    Heatmaps make the shapes legible at a glance. Watch for three things:

    1. `A` is arbitrary — no visible structure.
    2. `U` and `Vt` are **orthogonal** matrices: their columns are unit vectors at right angles.
    3. `S` contains only non-negative, descending values. Those are the stretch factors, and they
       are the only numbers that carry "size" information; `U` and `Vt` carry only direction.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


@app.cell
def _(plt, sns):
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
    return


@app.cell
def _(A, S, U, Vt, np, plt, sns):
    S_diag = np.diag(S)

    _panels = [
        (A, "A   (original)", "rocket", None),
        (U, "U   (left singular vectors)", "vlag", 0.0),
        (S_diag, r"$\Sigma$   (singular values)", "rocket", None),
        (Vt, r"$V^\top$   (right singular vectors)", "vlag", 0.0),
    ]

    _fig, _axes = plt.subplots(1, 4, figsize=(19.5, 4.8), constrained_layout=True)
    for _ax, (_mat, _title, _cmap, _center) in zip(_axes, _panels):
        sns.heatmap(
            _mat,
            ax=_ax,
            cmap=_cmap,
            center=_center,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 9.5, "fontweight": "medium"},
            linewidths=1.4,
            linecolor="#fbfbf9",
            cbar=True,
            cbar_kws={"shrink": 0.72, "pad": 0.02},
        )
        _ax.set_title(_title, fontsize=11, pad=12)
        _ax.set_xlabel("")
        _ax.set_ylabel("")

    _fig.suptitle(
        r"$A = U\,\Sigma\,V^\top$"
        rf"   ·   A: {A.shape[0]}×{A.shape[1]}"
        rf"   ·   U: {U.shape[0]}×{U.shape[1]}"
        rf"   ·   Σ: {S.size} values"
        rf"   ·   Vᵀ: {Vt.shape[0]}×{Vt.shape[1]}",
        fontsize=14,
        fontweight="bold",
        color="#1f2937",
    )
    _fig
    return (S_diag,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The singular value spectrum

    Sort order matters. Because $\sigma_1 \ge \sigma_2 \ge \dots$, a few singular values usually
    dominate, and the **cumulative energy** $\sum_{k \le i} \sigma_k^2$ shows how much of the
    matrix one rank-$i$ approximation already captures. That is exactly the justification behind
    PCA and truncated SVD compression: dropping the tail costs little, because the tail is small.

    The ratio $\sigma_1 / \sigma_{\min}$ is the **condition number** — a large value means the
    matrix is close to singular and inverting it will amplify noise.
    """)
    return


@app.cell
def _(A, S, np, plt, sns):
    import pandas as pd

    _sv = pd.DataFrame(
        {
            "component": [rf"$\sigma_{i + 1}$" for i in range(S.size)],
            "singular value": np.asarray(S, dtype=float),
        }
    )
    _sv["energy %"] = 100 * _sv["singular value"] ** 2 / (_sv["singular value"] ** 2).sum()
    _sv["cumulative %"] = _sv["energy %"].cumsum()

    _fig, _ax = plt.subplots(figsize=(10.0, 4.4), constrained_layout=True)
    sns.barplot(
        data=_sv,
        x="component",
        y="singular value",
        hue="component",
        palette="crest",
        legend=False,
        width=0.52,
        saturation=0.95,
        edgecolor="#fbfbf9",
        linewidth=2,
        ax=_ax,
    )
    for _patch, _val in zip(_ax.patches, _sv["singular value"]):
        _ax.annotate(
            f"{_val:.3f}",
            (_patch.get_x() + _patch.get_width() / 2, _val),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#1f2937",
        )
    _ax.set_ylim(0.0, float(S.max()) * 1.22)
    _ax.set_xlabel("")
    _ax.set_ylabel(r"singular value $\sigma_k$")
    _ax.set_title(
        "Singular values   ·   "
        rf"$\kappa(A) = {np.linalg.cond(A):.1f}$"
        r"   ·   bars are magnitude, red line is cumulative energy",
        fontsize=11,
        pad=12,
    )
    sns.despine(ax=_ax)

    _ax_energy = _ax.twinx()
    _ax_energy.plot(
        range(len(_sv)),
        _sv["cumulative %"],
        color="#e11d48",
        marker="o",
        markersize=6,
        lw=2.2,
        zorder=5,
    )
    _ax_energy.set_ylim(0, 105)
    _ax_energy.set_ylabel("cumulative energy  (%)", color="#e11d48")
    _ax_energy.tick_params(axis="y", colors="#e11d48")
    _ax_energy.grid(False)
    sns.despine(ax=_ax_energy, right=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building the SVD by hand

    The factorization is not magic. For a square, symmetric, positive definite matrix it collapses
    into one clean recipe:

    1. Form $B = A A^\top$. This is symmetric positive semidefinite, so `np.linalg.eigh` gives
       **real** eigenvalues and **orthonormal** eigenvectors — no complex arithmetic, no ordering
       surprises.
    2. Sort the eigenvalues descending and take square roots: $\sigma_i = \sqrt{\lambda_i}$. These
       are the singular values, and they come out automatically non-negative.
    3. Recover the left singular vectors from $A v_i = \sigma_i u_i$, i.e.
       $u_i = \dfrac{A v_i}{\sigma_i}$.

    We will do exactly that for $A_1 = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$, a matrix with
    eigenvalues $4$ and $2$ and therefore singular values $4$ and $2$ — the symmetric case, where
    singular values and eigenvalues coincide up to sign.
    """)
    return


@app.cell
def _(np):
    A1 = np.array([[3, 1], [1, 3]], dtype=float)
    A1
    return (A1,)


@app.cell
def _(A1):
    B = A1 @ A1.T
    B
    return (B,)


@app.cell
def _(B, np):
    eigenvalues, V = np.linalg.eigh(B)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    eigenvalues, V
    return V, eigenvalues


@app.cell
def _(eigenvalues, np):
    Sv = np.sqrt(np.maximum(eigenvalues, 0))
    Sv
    return (Sv,)


@app.cell
def _(A1, Sv, V, np):
    U1 = (A1 @ V) / Sv[np.newaxis, :]
    U1
    return (U1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the factorization is doing, geometrically

    A linear map turns the unit circle into an ellipse. The **right** singular vectors $v_i$ are
    the directions *in the input space* that get mapped onto the ellipse axes, the **left**
    singular vectors $u_i$ are those axes, and the singular values $\sigma_i$ are the semi-axis
    lengths.

    Read the two panels as the same story from two angles:

    - **Left:** the unit circle becomes an ellipse. Arrows of length $\sigma_i$ point along the
      ellipse axes $u_i$; dotted lines mark the pre-image directions $v_i$.
    - **Right:** the same map applied to a grid. Squares are sent to parallelograms, and the area
      scale factor is $\sigma_1 \sigma_2 = |\det A_1|$.
    """)
    return


@app.cell
def _(A1, Sv, U1, V, np, plt, sns):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    INK, MUTED = "#1f2937", "#94a3b8"
    TEAL, TEAL_LIGHT = "#0f766e", "#99f6e4"
    SKY = "#bfdbfe"
    VEC = ["#e11d48", "#d97706"]
    arrow = dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=15)
    dashed = (0, (3, 3))

    _fig, (ax_circle, ax_grid) = plt.subplots(
        1, 2, figsize=(15.5, 6.6), constrained_layout=True
    )

    # ---------------- left: unit circle → ellipse ----------------
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse = circle @ A1.T

    ax_circle.fill(ellipse[:, 0], ellipse[:, 1], color=TEAL_LIGHT, alpha=0.55, zorder=1)
    ax_circle.plot(ellipse[:, 0], ellipse[:, 1], color=TEAL, lw=2.2, zorder=2)
    ax_circle.plot(circle[:, 0], circle[:, 1], color=MUTED, lw=1.3, ls=dashed, zorder=1)

    for i, sigma in enumerate(Sv):
        u, v = U1[:, i], V[:, i]
        tip = sigma * u
        perp = np.array([-u[1], u[0]])

        # pre-image direction in the input space
        ax_circle.plot(
            [-1.55 * v[0], 1.55 * v[0]], [-1.55 * v[1], 1.55 * v[1]],
            color=VEC[i], lw=1.2, ls=":", alpha=0.8, zorder=3,
            label="_nolegend_",
        )
        ax_circle.annotate(
            "", xy=tip, xytext=(0, 0),
            arrowprops={**arrow, "color": VEC[i], "lw": 2.4}, zorder=4,
        )
        ax_circle.text(
            *(sigma + 0.35) * u + 0.18 * perp,
            rf"$\sigma_{i + 1}={sigma:.1f}$",
            color=VEC[i], fontsize=10.5, fontweight="semibold",
            ha="left" if u[0] > 0 else "right",
            va="bottom" if u[1] > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#fbfbf9",
                      edgecolor="none", alpha=0.85),
            zorder=5,
        )

    lim = 1.25 * float(Sv.max())
    ax_circle.set_xlim(-lim, lim)
    ax_circle.set_ylim(-lim, lim)
    ax_circle.set_aspect("equal", adjustable="box")
    ax_circle.axhline(0, color="#e2e8f0", lw=1.2, zorder=0)
    ax_circle.axvline(0, color="#e2e8f0", lw=1.2, zorder=0)
    ax_circle.grid(False)
    ax_circle.set_xticks([])
    ax_circle.set_yticks([])
    ax_circle.set_title(
        "Unit circle → ellipse:  semi-axes are $\\sigma_i$ along $u_i$",
        fontsize=11.5,
        pad=12,
    )
    ax_circle.legend(
        handles=[
            Line2D([], [], color=MUTED, ls=dashed, lw=1.3, label="unit circle"),
            Line2D([], [], color=TEAL, lw=2.2, label=r"$A_1\cdot$unit circle"),
            Line2D([], [], color=VEC[0], lw=2.4, label=rf"$u_1$, length $\sigma_1={Sv[0]:.1f}$"),
            Line2D([], [], color=VEC[1], lw=2.4, label=rf"$u_2$, length $\sigma_2={Sv[1]:.1f}$"),
            Line2D([], [], color=MUTED, ls=":", lw=1.2, label=r"pre-image directions $v_i$"),
        ],
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        borderpad=0.7,
    )
    sns.despine(ax=ax_circle, left=True, bottom=True, top=True, right=True)

    # ---------------- right: grid and its image under A1 ----------------
    span = 2.0
    t = np.linspace(-span, span, 140)

    def seg(ax, pts, **kwargs):
        ax.plot(pts[:, 0], pts[:, 1], **kwargs)

    for k in range(-3, 4):
        if k == 0:
            continue
        # original grid
        seg(ax_grid, np.column_stack([np.full_like(t, k), t]),
            color=MUTED, lw=1.0, ls=dashed, alpha=0.65, zorder=1)
        seg(ax_grid, np.column_stack([t, np.full_like(t, k)]),
            color=MUTED, lw=1.0, ls=dashed, alpha=0.65, zorder=1)
        # image grid
        seg(ax_grid, np.column_stack([np.full_like(t, k), t]) @ A1.T,
            color=TEAL, lw=1.1, alpha=0.45, zorder=2)
        seg(ax_grid, np.column_stack([t, np.full_like(t, k)]) @ A1.T,
            color=TEAL, lw=1.1, alpha=0.45, zorder=2)

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    ax_grid.fill(square[:, 0], square[:, 1], color=SKY, alpha=0.55, zorder=0)
    img_square = square @ A1.T
    ax_grid.fill(
        img_square[:, 0], img_square[:, 1],
        color=TEAL_LIGHT, alpha=0.75, edgecolor=TEAL, lw=1.2, zorder=0,
    )

    for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        ax_grid.annotate("", xy=e, xytext=(0, 0),
                         arrowprops={**arrow, "color": MUTED, "lw": 1.7}, zorder=4)
        ax_grid.annotate("", xy=A1 @ e, xytext=(0, 0),
                         arrowprops={**arrow, "color": TEAL, "lw": 2.2}, zorder=5)

    g_lim = 1.2 * float(np.abs(img_square).max())
    ax_grid.set_xlim(-g_lim, g_lim)
    ax_grid.set_ylim(-g_lim, g_lim)
    ax_grid.set_aspect("equal", adjustable="box")
    ax_grid.axhline(0, color="#cbd5e1", lw=1.4, zorder=1)
    ax_grid.axvline(0, color="#cbd5e1", lw=1.4, zorder=1)
    ax_grid.grid(False)
    ax_grid.set_xlabel("x")
    ax_grid.set_ylabel("y")
    ax_grid.set_title(
        "The grid, before and after   ·   "
        rf"area scale $=\sigma_1\sigma_2={Sv[0] * Sv[1]:.1f}$ "
        rf"$=|\det A_1|={abs(np.linalg.det(A1)):.1f}$",
        fontsize=11.5,
        pad=12,
    )
    ax_grid.legend(
        handles=[
            Line2D([], [], color=MUTED, ls=dashed, lw=1.2, label="original grid"),
            Line2D([], [], color=TEAL, lw=1.4, label=r"$A_1\cdot$grid"),
            Patch(facecolor=SKY, alpha=0.7, label="unit square"),
            Patch(facecolor=TEAL_LIGHT, alpha=0.85, label=r"$A_1\cdot$unit square"),
        ],
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        borderpad=0.7,
    )
    sns.despine(ax=ax_grid)

    _fig.suptitle(
        "A1 = [[3, 1], [1, 3]]   —   rotate, stretch, rotate back",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The newly constructed SVD

    Putting the three pieces back together, $U_1 \Sigma V^\top$ should reproduce $A_1$ to machine
    precision. The heatmaps below compare the original, the reconstruction, and the absolute
    residual — the residual panel should be visually empty, which is the point: the SVD is exact,
    not an approximation. Truncation is a deliberate choice you make afterwards.
    """)
    return


@app.cell
def _(A1, Sv, U1, V, np):
    reconstructed = U1 @ np.diag(Sv) @ V.T

    checks = {
        "max |A1 − UΣVᵀ|": float(np.abs(A1 - reconstructed).max()),
        "max |UᵀU − I|": float(np.abs(U1.T @ U1 - np.eye(U1.shape[1])).max()),
        "max |VᵀV − I|": float(np.abs(V.T @ V - np.eye(V.shape[1])).max()),
        "σ vs. numpy": float(np.abs(Sv - np.linalg.svd(A1, compute_uv=False)).max()),
    }
    return (checks, reconstructed)


@app.cell
def _(A1, reconstructed, np, plt, sns):
    residual = np.abs(A1 - reconstructed)
    _base = dict(
        annot=True,
        linewidths=1.6,
        linecolor="#fbfbf9",
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.72, "pad": 0.02},
    )
    _panels = [
        (
            A1,
            r"$A_1$  (original)",
            "rocket",
            dict(fmt=".2f", annot_kws={"fontsize": 11, "fontweight": "medium"}),
        ),
        (
            reconstructed,
            r"$U_1 \Sigma V^\top$  (rebuilt)",
            "rocket",
            dict(fmt=".2f", annot_kws={"fontsize": 11, "fontweight": "medium"}),
        ),
        (
            residual,
            "absolute residual   (note the $10^{-16}$ scale)",
            "YlOrRd",
            dict(
                fmt=".2e",
                vmax=1e-12,
                annot_kws={"fontsize": 10.5, "color": "#334155"},
            ),
        ),
    ]

    _fig, _axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for _ax, (_mat, _title, _cmap, _opts) in zip(_axes, _panels):
        sns.heatmap(_mat, ax=_ax, cmap=_cmap, **_base, **_opts)
        _ax.set_title(_title, fontsize=11.5, pad=12)
        _ax.set_xlabel("")
        _ax.set_ylabel("")

    _fig.suptitle(
        r"$A_1 = U_1\,\Sigma\,V^\top$   —   exact to machine precision",
        fontsize=14,
        fontweight="bold",
        color="#1f2937",
    )
    _fig
    return (residual,)


@app.cell
def _(checks, mo):
    mo.ui.table(
        [{"check": name, "value": value} for name, value in checks.items()],
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where this shows up

    - **PCA** — eigendecomposition of the covariance matrix is the SVD of the centred data.
    - **Low-rank approximation** — truncating after $k$ terms gives the best rank-$k$ matrix in
      Frobenius norm (Eckart–Young).
    - **Moore–Penrose pseudoinverse** — $A^{+} = V \Sigma^{+} U^\top$, which inverts only the
      directions that carry real information.
    - **Numerical stability** — the condition number $\sigma_1 / \sigma_n$ tells you how much a
      linear solve will amplify error.
    """)
    return


if __name__ == "__main__":
    app.run()

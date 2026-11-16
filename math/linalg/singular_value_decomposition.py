import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup — `marimo`, `numpy`

    > `import marimo as mo` · `import numpy as np`

    Every cell below builds on two libraries:

    | import | role |
    |---|---|
    | `marimo as mo` | the notebook runtime. `mo.md(...)` renders narrative cells like this one; `mo.ui.table(...)` renders the interactive check table at the end. |
    | `numpy as np` | all of the linear algebra. Every decomposition, product, and norm is plain `numpy` — no scipy, no custom solvers. |

    Imported once here, then used bare as `mo.` and `np.` throughout.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.random.default_rng`

    > `np.random.default_rng(seed=None)`

    Constructs a **Generator**, numpy's modern random source and the replacement for the legacy global `np.random.seed` state machine.

    | argument | meaning |
    |---|---|
    | `seed` | int, sequence, or `SeedSequence`. The same seed always yields the same stream — that is what makes the notebook reproducible. |

    **Returns** a `Generator` exposing `.random()`, `.normal()`, `.integers()`, `.permutation()`, and friends.

    **In this notebook:** `seed=42` fixes the example matrix, so every reader sees the identical `A`, the identical singular values, and identical figures.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Generator.random`

    > `rng.random(size=None, dtype=np.float64)`

    Draws samples from the **uniform distribution on `[0, 1)`**.

    | argument | meaning |
    |---|---|
    | `size` | output shape — `(3, 3)` here; `None` returns a single float |

    **In this notebook:** `rng.random(size=(3, 3))` gives an arbitrary, structureless matrix. That is deliberate — the SVD exists for *any* matrix, so the example is deliberately generic rather than a matrix chosen to look nice.
    """)
    return


@app.cell
def _(rng):
    A = rng.random(size=(3, 3))
    A
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.linalg.svd`

    > `np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)`

    The singular value decomposition `A = U Σ Vᵀ`.

    | argument | meaning |
    |---|---|
    | `a` | the `(M, N)` input — any shape, any rank |
    | `full_matrices` | `True`: `U` is `M×M`, `Vh` is `N×N`. `False`: economy shapes `M×K` and `K×N`, with `K = min(M, N)` |
    | `compute_uv` | `False` returns only `S`, skipping the orthogonal factors |
    | `hermitian` | `True` assumes `A = Aᴴ`, enabling a faster real-only path |

    **Returns** `U, S, Vh`, with `S` in **descending** order and `A == U @ np.diag(S) @ Vh`.

    **In this notebook:** `full_matrices=False` keeps the factors at their minimal shapes, so nothing is padded out with meaningless zeros.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting stack — `matplotlib.pyplot`, `seaborn`

    > `import matplotlib.pyplot as plt` · `import seaborn as sns`

    | import | role |
    |---|---|
    | `matplotlib.pyplot as plt` | the canvas: `subplots`, `plot`, `fill`, `annotate`, and the trailing `_fig` expression that renders a cell's output |
    | `seaborn as sns` | the styling layer: themes, palettes, `heatmap`, `barplot`, and `despine` for quieter axes |

    Split across two cells because the next one configures the theme before any figure is drawn.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `sns.set_theme`

    > `sns.set_theme(context="notebook", style="whitegrid", palette="deep", rc=None)`

    Sets seaborn's global theme in one call: matplotlib defaults, grid style, and default palette together.

    | argument | meaning |
    |---|---|
    | `context` | element scale — `"notebook"`, `"talk"`, `"paper"`; controls font sizes and line widths |
    | `style` | axes styling; `"whitegrid"` is a white background with a light grid |
    | `palette` | default categorical palette, here `"deep"` |
    | `rc` | dict of raw matplotlib rcParams merged on top — the escape hatch for per-element colors |

    **In this notebook:** `rc` pins one warm off-white (`#fbfbf9`) for figure, axes, and saved-image backgrounds, plus muted slate text and grid lines. Every later plot inherits that palette, so the figures stay visually consistent without repeating style code.
    """)
    return


@app.cell
def _(sns):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.diag` and `sns.heatmap`

    > `np.diag(v, k=0)` · `sns.heatmap(data, annot=False, fmt=".2g", cmap=None, center=None)`

    **`np.diag`** is overloaded by input rank: given a 1-D array it *builds* a matrix with that array on the `k`-th diagonal; given a 2-D array it *extracts* one. Here `np.diag(S)` promotes the singular-value vector into the diagonal `Σ`.

    **`sns.heatmap`** draws a matrix as a color-coded grid.

    | argument | meaning |
    |---|---|
    | `annot`, `fmt` | print each value in its cell, formatted as `.2f` |
    | `cmap` | `"rocket"` is sequential (dark = large); `"vlag"` diverges blue↔red around `center` |
    | `center` | value pinned to the middle of a diverging map — `0.0` makes signed entries readable |
    | `linewidths`, `linecolor` | cell borders, chosen to match the background so cells read as separate tiles |
    | `cbar_kws` | forwarded to the colorbar (`shrink`, `pad`) |

    **In this notebook:** four panels side by side, and the lesson is in the shapes — `A` is arbitrary, `U` and `Vᵀ` are orthogonal (columns are unit vectors at right angles), `Σ` is diagonal with descending non-negative entries.
    """)
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
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `pandas.DataFrame`, `sns.barplot`, `np.linalg.cond`

    > `pd.DataFrame(data)` · `sns.barplot(data, x, y, hue=None, palette=None, width=0.8)` · `np.linalg.cond(x, p=None)`

    **`pd.DataFrame`** — a labeled table. The singular values, each one's share of the energy, and the running cumulative share are assembled into one frame; assignments like `_sv["energy %"] = ...` are vectorized numpy arithmetic. Seaborn then reads columns by *name*, which keeps the plotting call readable.

    **`sns.barplot`** — draws one bar per category, aggregating with a mean by default.

    | argument | meaning |
    |---|---|
    | `x`, `y` | category and value columns |
    | `hue` | color grouping; reusing the `x` column gives each bar its own palette step |
    | `width`, `saturation` | bar geometry and color intensity |
    | `edgecolor`, `linewidth` | outline around each bar — drawn in the background color for visual separation |
    | `legend=False` | suppresses the redundant hue legend |

    **`np.linalg.cond`** — the 2-norm condition number `σ_max / σ_min`. It is printed in the title: a large value means the matrix is nearly singular, so a linear solve would amplify noise.

    **In this notebook:** bars are magnitude, the red twin-axis line is cumulative energy, and together they make the case for truncated SVD — a couple of components already carry nearly all of the energy.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.array`

    > `np.array(object, dtype=None, copy=True)`

    Wraps a Python nested list as an `ndarray`; the nesting becomes the shape.

    | argument | meaning |
    |---|---|
    | `object` | nested lists — `[[3, 1], [1, 3]]` becomes shape `(2, 2)` |
    | `dtype` | element type; `float` here, so the later division and `sqrt` stay real and exact |

    **In this notebook:** `A1` is symmetric and positive definite with eigenvalues `4` and `2`. Deliberately tiny — small enough to verify the hand-built SVD with mental arithmetic.
    """)
    return


@app.cell
def _(np):
    A1 = np.array([[3, 1], [1, 3]], dtype=float)
    A1
    return (A1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `@` (matmul) and `.T`

    > `A1 @ A1.T` · `a.T`

    **`@`** is `np.matmul`: the true matrix product, not elementwise. Shape rule: `(m, k) @ (k, n) → (m, n)`.

    **`.T`** is a transpose view — rows become columns, no copy of the data.

    **In this notebook:** `B = A1 A1ᵀ` is symmetric, because transposing the product swaps the two factors, and it is positive semidefinite. Those two facts are what make step 2 of the hand recipe safe: `eigh` returns real, non-negative eigenvalues, so the square root never sees a negative number and never produces a complex vector.
    """)
    return


@app.cell
def _(A1):
    B = A1 @ A1.T
    B
    return (B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.linalg.eigh` and `np.argsort`

    > `np.linalg.eigh(a, UPLO="L")` · `np.argsort(a)`

    **`np.linalg.eigh`** diagonalizes a **symmetric / Hermitian** matrix: `B = V Λ Vᵀ`, with guaranteed real eigenvalues and orthonormal eigenvectors. It is the correct routine for `A Aᵀ`; the general `np.linalg.eig` is slower and may return complex values even for a provably real problem.

    **Returns** `(eigenvalues, eigenvectors)`, with eigenvalues **ascending** and `eigenvectors[:, i]` belonging to `eigenvalues[i]`.

    **`np.argsort(a)`** returns the indices that would sort `a`, without touching `a` itself; `[::-1]` flips them to descending.

    **In this notebook:** `eigh` hands back ascending order while the SVD convention is descending, so the eigenvalues *and* the matching columns of `V` are reordered with the same index array. Reordering one without the other would silently shred the eigenvector pairing.
    """)
    return


@app.cell
def _(B, np):
    eigenvalues, V = np.linalg.eigh(B)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    eigenvalues, V
    return V, eigenvalues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.sqrt` and `np.maximum`

    > `np.sqrt(x)` · `np.maximum(x1, x2)`

    **`np.sqrt`** is elementwise and returns `nan` for negative input. **`np.maximum`** takes the elementwise larger of two operands — the clip used here.

    **Why both:** the eigenvalues of `A Aᵀ` are non-negative in exact arithmetic, but floating-point round-off can hand back a tiny `-2e-16`. `np.maximum(eigenvalues, 0)` clamps that to zero before the square root, turning a `nan` outbreak into an honest `0.0`.

    **In this notebook:** `σᵢ = √λᵢ` — the stretch factors. For this symmetric example they come out as exactly `4.0` and `2.0`, matching `np.linalg.svd(A1)`.
    """)
    return


@app.cell
def _(eigenvalues, np):
    Sv = np.sqrt(np.maximum(eigenvalues, 0))
    Sv
    return (Sv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Broadcasting and `np.newaxis`

    > `(A1 @ V) / Sv[np.newaxis, :]`

    **`np.newaxis`** inserts a length-1 axis, turning the `(2,)` vector `Sv` into a `(1, 2)` row. Broadcasting then stretches that row across both rows of the product: `(2, 2) / (1, 2)` divides column `j` by `Sv[j]`.

    That is exactly `uᵢ = A vᵢ / σᵢ` for every column at once, no Python loop.

    **Watch out:** without `[np.newaxis, :]` numpy aligns the vector with the *rows* instead and produces a silently wrong result — the classic broadcasting bug in linear algebra code.

    **In this notebook:** this one line recovers the left singular vectors from `A1 V = U Σ`.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Line2D`, `Patch`, and `Axes.annotate`

    > `from matplotlib.lines import Line2D` · `from matplotlib.patches import Patch` · `ax.annotate("\", xy=head, xytext=tail, arrowprops=...)`

    Three drawing primitives, all fed plain numpy arrays:

    | primitive | use here |
    |---|---|
    | `Line2D` | a *detached* line object, created empty with `Line2D([], [])` — never drawn, it exists only to hand the legend a handle with the right color and dash style |
    | `Patch` | a filled polygon carrier; `Patch(facecolor=...)` builds legend swatches for the shaded unit square and its image |
    | `ax.annotate` + `arrowprops` | draws the vectors: empty text, `xy` is the arrow head, `xytext` the tail |
    | `ax.fill`, `ax.plot` | the ellipse, the grid, and the transformed square |

    **In this notebook:** `arrow` and `dashed` are shared dicts so every arrow and dotted line is styled identically. `circle @ A1.T` pushes all 400 points of the unit circle through the map as a single `(400, 2)` array — one matmul, no loop over angles.
    """)
    return


@app.cell(expand_output=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reassembly — `np.diag`, `np.eye`, `np.abs`

    > `np.diag(Sv)` · `np.eye(n)` · `np.abs(x)`

    `U1 @ np.diag(Sv) @ V.T` replays the factorization literally: rotate by `Vᵀ`, stretch by `Σ`, rotate by `U1`.

    The `checks` dict then measures four independent errors:

    | key | what it proves |
    |---|---|
    | `max abs(A1 - U S Vt)` | the decomposition reconstructs the matrix |
    | `max abs(UᵀU - I)` | `U` is orthogonal (`np.eye` builds the identity to compare against) |
    | `max abs(VᵀV - I)` | `V` is orthogonal |
    | `sigma vs. numpy` | the hand-built values match `np.linalg.svd(A1, compute_uv=False)` |

    `np.abs(...).max()` is the infinity norm of the residual: the worst single entry, which is the strictest simple summary of how wrong the reconstruction is.

    **In this notebook:** every value should land near `1e-16`. That is not "close enough" — it means the SVD is exact up to floating-point representation.
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
    return checks, reconstructed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading differences with `sns.heatmap`

    > `sns.heatmap(data, annot=True, fmt=".2e", vmax=..., cmap="YlOrRd", square=True)`

    The same routine as the earlier panel, configured for a different job — reading *differences* rather than values:

    | setting | why |
    |---|---|
    | `cmap="rocket"` on `A1` and on the rebuild | identical scale, so the two panels are directly comparable |
    | `cmap="YlOrRd"` on the residual | white means "nothing", orange means error |
    | `fmt=".2e"` | residuals sit at `1e-16`; fixed-point `.2f` would print `0.00` in every cell |
    | `vmax=1e-12` | caps the color scale so one round-off blip cannot wash out the rest |
    | `square=True` | keeps the 2×2 tiles square instead of stretching them |

    **In this notebook:** the third panel is the punchline — a grid of numbers at the `1e-16` scale, which reads as visual silence.
    """)
    return


@app.cell
def _(A1, np, plt, reconstructed, sns):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `mo.ui.table`

    > `mo.ui.table(data, selection="multi", pagination=True, show_column_summaries=True)`

    Renders a list of dicts (or a DataFrame) as a **reactive, sortable HTML table** inside the notebook, not a static printout.

    | argument | meaning |
    |---|---|
    | `selection` | `"multi"`, `"single"`, or `None`; `None` drops the checkboxes because nothing here consumes a selection |
    | `show_column_summaries` | hides the min/mean/max footer — noise for four scalar checks |
    | `show_data_types` | hides the dtype row, which would just say `float` four times |

    **In this notebook:** the `checks` dict from the previous cell becomes four sortable rows, so the accuracy evidence is readable at a glance instead of buried in a printed dict.
    """)
    return


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

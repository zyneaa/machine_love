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
    rng = np.random.default_rng(42)
    return (rng,)


@app.cell
def _(np, rng):
    values = rng.integers(1, 10, size=10)
    D = np.diag(values)
    D
    return (D,)


@app.cell
def _(D, np):
    np.prod(np.diagonal(D))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finding the det of 0 in multiple Ds
    """)
    return


@app.cell
def _(np, rng):
    for i in range(100_000):
        val = rng.integers(1, 10, size=10)
        A = np.diag(val)
        if np.prod(np.diag(A)) == 0:
            print(A)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B as a geometric transformation

    A linear map sends the plane to itself. The **eigenvectors** are the special
    directions that come out pointing the same way; the **eigenvalues** are how much
    they stretch. Every other direction gets rotated *and* stretched.

    - **Left:** the original grid, with the image of that grid under `B` drawn on top.
    - **Top right:** the unit circle becomes an ellipse whose semi-axes are `|λ|`
      along the eigen-directions.
    - **Bottom right:** the eigenvalues, i.e. the stretch factors.

    `B` below is symmetric and positive definite, so the eigenvalues are real and the
    eigenvectors are orthogonal — the cleanest case to see. Swap in any 2×2 matrix
    (with real eigenvalues) to explore.
    """)
    return


@app.cell
def _(np):
    B = np.array([[2.0, 1.0], [1.0, 2.0]])
    return (B,)


@app.cell
def _(B, np):
    lam_B, V_B = np.linalg.eig(B)
    order = np.argsort(lam_B)[::-1]  # biggest stretch first
    lam_B, V_B = lam_B[order], V_B[:, order]
    for j in range(V_B.shape[1]):  # fix sign so arrows point into the right half-plane
        if V_B[0, j] < 0:
            V_B[:, j] *= -1
    lam_B, V_B
    return V_B, lam_B


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    return pd, plt, sns


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
def _(np):
    def b_line(ax, start, end, transform, extend=2.5, **kwargs):
        """Draw the image of the line through start→end under a linear map.

        `extend` pushes the drawn segment past both end points so the image runs
        off the edge of the axes instead of stopping in mid-air.
        """
        t = np.linspace(-extend, 1.0 + extend, 140)[:, None]
        pts = start[None, :] + t * (end - start)[None, :]
        pts = pts @ transform.T
        ax.plot(pts[:, 0], pts[:, 1], **kwargs)

    return (b_line,)


@app.cell
def _(B, V_B, b_line, lam_B, np, pd, plt, sns):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    INK, MUTED = "#1f2937", "#94a3b8"
    TEAL, TEAL_LIGHT = "#0f766e", "#99f6e4"
    SKY = "#bfdbfe"
    VEC = ["#e11d48", "#d97706"]

    fig = plt.figure(figsize=(15.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], height_ratios=[1.15, 1.0])
    ax_plane = fig.add_subplot(gs[:, 0])
    ax_circle = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, 1])

    # ---------------- left: the grid and its image under B ----------------
    span, dashed = 2.0, (0, (3, 3))
    for k in range(-3, 4):
        if k == 0:
            continue
        ax_plane.plot([k, k], [-span, span], color=MUTED, lw=1.0, ls=dashed, alpha=0.7, zorder=1)
        ax_plane.plot([-span, span], [k, k], color=MUTED, lw=1.0, ls=dashed, alpha=0.7, zorder=1)
        b_line(
            ax_plane, np.array([k, -span]), np.array([k, span]), B,
            color=TEAL, lw=1.1, alpha=0.4, zorder=2,
        )
        b_line(
            ax_plane, np.array([-span, k]), np.array([span, k]), B,
            color=TEAL, lw=1.1, alpha=0.4, zorder=2,
        )

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    ax_plane.fill(square[:, 0], square[:, 1], color=SKY, alpha=0.55, zorder=0)
    img_square = square @ B.T
    ax_plane.fill(
        img_square[:, 0], img_square[:, 1],
        color=TEAL_LIGHT, alpha=0.7, edgecolor=TEAL, lw=1.2, zorder=0,
    )

    arrow = dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=15)
    for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        ax_plane.annotate(
            "", xy=e, xytext=(0, 0),
            arrowprops={**arrow, "color": MUTED, "lw": 1.7}, zorder=4,
        )
        ax_plane.annotate(
            "", xy=B @ e, xytext=(0, 0),
            arrowprops={**arrow, "color": TEAL, "lw": 2.4}, zorder=5,
        )

    for idx, (lam, v) in enumerate(zip(lam_B, V_B.T)):
        perp = np.array([-v[1], v[0]])  # points away from the arrow
        d = v * span * 0.9
        ax_plane.plot(
            [-d[0], d[0]], [-d[1], d[1]],
            color=VEC[idx], lw=1.2, ls=":", alpha=0.85, zorder=3,
            label="_nolegend_",
        )
        tip = lam * v
        ax_plane.annotate(
            "", xy=tip, xytext=(0, 0),
            arrowprops={**arrow, "color": VEC[idx], "lw": 2.6}, zorder=6,
        )
        label_pos = (max(lam, 1.0) + 0.75) * v + 0.25 * perp
        ax_plane.text(
            *label_pos,
            rf"$v_{idx + 1}$  ($\lambda_{idx + 1}={lam:.1f}$)",
            color=VEC[idx], fontsize=10, fontweight="semibold",
            ha="left" if v[0] > 0 else "right",
            va="bottom" if v[1] > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#fbfbf9", edgecolor="none", alpha=0.8),
            zorder=6,
        )

    corners = np.array([[x, y] for x in (-span, span) for y in (-span, span)]) @ B.T
    lim = 1.12 * np.abs(corners).max()
    ax_plane.set_xlim(-lim, lim)
    ax_plane.set_ylim(-lim, lim)
    ax_plane.set_aspect("equal", adjustable="box")
    ax_plane.axhline(0, color="#cbd5e1", lw=1.4, zorder=1)
    ax_plane.axvline(0, color="#cbd5e1", lw=1.4, zorder=1)
    ax_plane.grid(False)
    ax_plane.set_xlabel("x")
    ax_plane.set_ylabel("y")
    ax_plane.set_title(
        f"B = [[{B[0, 0]:g}, {B[0, 1]:g}], [{B[1, 0]:g}, {B[1, 1]:g}]]"
        "   —   original grid vs. its image",
        fontsize=12,
        pad=12,
    )
    ax_plane.legend(
        handles=[
            Line2D([], [], color=MUTED, ls=dashed, lw=1.2, label="original grid"),
            Line2D([], [], color=TEAL, lw=1.4, label=r"$B\cdot$grid"),
            Patch(facecolor=SKY, alpha=0.7, label="unit square"),
            Patch(facecolor=TEAL_LIGHT, alpha=0.8, label=r"$B\cdot$unit square"),
            Line2D([], [], color=VEC[0], lw=2.4, label=rf"$v_1$, stretch {lam_B[0]:.2f}"),
            Line2D([], [], color=VEC[1], lw=2.4, label=rf"$v_2$, stretch {lam_B[1]:.2f}"),
        ],
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        borderpad=0.7,
    )

    # ---------------- top right: unit circle → ellipse ----------------
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse = circle @ B.T
    ax_circle.fill(ellipse[:, 0], ellipse[:, 1], color=TEAL_LIGHT, alpha=0.55, zorder=1)
    ax_circle.plot(ellipse[:, 0], ellipse[:, 1], color=TEAL, lw=2.2, zorder=2)
    ax_circle.plot(circle[:, 0], circle[:, 1], color=MUTED, lw=1.3, ls=dashed, zorder=1)

    for idx, (lam, v) in enumerate(zip(lam_B, V_B.T)):
        perp = np.array([-v[1], v[0]])
        tip = lam * v
        ax_circle.annotate(
            "", xy=tip, xytext=(0, 0),
            arrowprops={**arrow, "color": VEC[idx], "lw": 2.4}, zorder=4,
        )
        ax_circle.text(
            *(abs(lam) + 0.5) * v + 0.08 * perp, rf"$|\lambda_{idx + 1}|={abs(lam):.1f}$",
            color=VEC[idx], fontsize=9.5, fontweight="semibold",
            ha="center", va="center", zorder=4,
        )

    c_lim = 1.18 * float(np.abs(lam_B).max())
    ax_circle.set_xlim(-c_lim, c_lim)
    ax_circle.set_ylim(-c_lim, c_lim)
    ax_circle.set_aspect("equal", adjustable="box")
    ax_circle.axhline(0, color="#e2e8f0", lw=1.2, zorder=0)
    ax_circle.axvline(0, color="#e2e8f0", lw=1.2, zorder=0)
    ax_circle.grid(False)
    ax_circle.set_title(
        "Unit circle → ellipse: eigenvalues are the stretch factors",
        fontsize=11,
        pad=10,
    )
    sns.despine(ax=ax_circle, left=True, bottom=True)
    ax_circle.set_xticks([])
    ax_circle.set_yticks([])

    # ---------------- bottom right: eigenvalue spectrum ----------------
    labels = [r"$v_1$", r"$v_2$"]
    spec = pd.DataFrame(
        {"eigenvector": labels, "eigenvalue": np.asarray(lam_B, dtype=float)}
    )
    sns.barplot(
        data=spec,
        x="eigenvector",
        y="eigenvalue",
        hue="eigenvector",
        palette={labels[0]: VEC[0], labels[1]: VEC[1]},
        legend=False,
        width=0.5,
        saturation=0.95,
        edgecolor="#fbfbf9",
        linewidth=2,
        ax=ax_spec,
    )
    for patch, eig_val in zip(ax_spec.patches, spec["eigenvalue"]):
        ax_spec.annotate(
            f"{eig_val:.2f}",
            (patch.get_x() + patch.get_width() / 2, eig_val),
            xytext=(0, 7 if eig_val >= 0 else -16),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    ref = ax_spec.axhline(
        1.0, color=MUTED, ls=":", lw=1.4, zorder=1,
        label=r"$\lambda=1$: length unchanged",
    )
    ax_spec.legend(
        handles=[ref], loc="upper right",
        frameon=False, fontsize=8.5, handlelength=1.8,
    )
    ax_spec.set_ylim(
        min(0.0, float(np.min(lam_B)) * 1.3),
        max(1.5, float(np.max(lam_B)) * 1.3),
    )
    ax_spec.set_xlabel("")
    ax_spec.set_ylabel("stretch factor $\\lambda$")
    ax_spec.set_title(
        rf"Eigenvalues   ·   $\det B = {np.linalg.det(B):.0f}$ (area scale)"
        rf"   ·   $\mathrm{{tr}}\, B = {np.trace(B):.0f}$",
        fontsize=11,
        pad=10,
    )
    sns.despine(ax=ax_spec)

    fig.suptitle(
        "Eigendecomposition of $B$: the directions that survive the transformation",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig
    return


if __name__ == "__main__":
    app.run()

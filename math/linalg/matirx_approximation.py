import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return mo, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Matrix Approximation with SVD

    Explore how **singular value decomposition (SVD)** approximates a matrix with a lower-rank version. Adjust the rank slider to trade detail for compression.

    The example matrix contains a rank-2 signal plus small noise. NumPy computes the decomposition; Seaborn and Matplotlib visualize the result.
    """)
    return


@app.cell
def _(np):
    rows, cols = 80, 60
    row_coords = np.linspace(0, 2 * np.pi, rows)
    col_coords = np.linspace(0, 2 * np.pi, cols)

    low_rank_signal = 3.0 * np.outer(
        np.sin(row_coords), np.cos(col_coords)
    ) + 1.5 * np.outer(np.cos(2 * row_coords), np.sin(3 * col_coords))
    rng = np.random.default_rng(7)
    matrix = low_rank_signal + rng.normal(0, 0.25, size=(rows, cols))

    matrix.shape, rows * cols
    return (matrix,)


@app.cell
def _(matrix, mo, np):
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)

    rank_control = mo.ui.slider(
        start=1,
        stop=len(singular_values),
        value=30,
        step=1,
        label="Approximation rank",
    )
    rank_control
    return U, Vt, rank_control, singular_values


@app.cell
def _(U, Vt, singular_values):
    U[5], singular_values, Vt
    return


@app.cell
def _(U, Vt, matrix, mo, np, rank_control, singular_values):
    selected_rank = rank_control.value
    approximation = (
        U[:, :selected_rank] * singular_values[:selected_rank]
    ) @ Vt[:selected_rank, :]
    residual = matrix - approximation
    retained_energy = np.sum(singular_values[:selected_rank] ** 2) / np.sum(
        singular_values**2
    )
    relative_error = np.linalg.norm(residual, ord="fro") / np.linalg.norm(
        matrix, ord="fro"
    )

    mo.md(
        f"Rank **{selected_rank}** retains **{retained_energy:.1%}** of singular-value energy; "
        f"relative Frobenius error: **{relative_error:.1%}**."
    )
    return approximation, residual, selected_rank


@app.cell
def _(approximation, matrix, plt, residual, selected_rank, sns):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    shared_min, shared_max = matrix.min(), matrix.max()

    sns.heatmap(
        matrix,
        ax=axes[0],
        cmap="viridis",
        vmin=shared_min,
        vmax=shared_max,
        cbar=False,
    )
    axes[0].set_title("Original matrix")

    sns.heatmap(
        approximation,
        ax=axes[1],
        cmap="viridis",
        vmin=shared_min,
        vmax=shared_max,
        cbar=False,
    )
    axes[1].set_title(f"Rank-{selected_rank} approximation")

    sns.heatmap(residual, ax=axes[2], cmap="vlag", center=0, cbar=False)
    axes[2].set_title("Residual")

    for ax in axes:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()
    fig
    return


@app.cell
def _(np, plt, selected_rank, singular_values, sns):
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 3.5))
    component_numbers = np.arange(1, len(singular_values) + 1)

    sns.lineplot(
        x=component_numbers,
        y=singular_values,
        marker="o",
        ax=ax_spectrum,
    )
    ax_spectrum.axvline(
        selected_rank,
        color="crimson",
        linestyle="--",
        label=f"Chosen rank: {selected_rank}",
    )
    ax_spectrum.set(
        title="Singular value spectrum",
        xlabel="Component",
        ylabel="Singular value",
    )
    ax_spectrum.legend()
    fig_spectrum.tight_layout()
    fig_spectrum
    return


if __name__ == "__main__":
    app.run()

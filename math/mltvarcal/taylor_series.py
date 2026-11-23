import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    ## Taylor Series Challenge

    Let \(f(x)=\sqrt{1+x}\), centered at \(a=0\).

    1. Find the degree-3 Maclaurin polynomial \(T_3(x)\).
    2. Use it to estimate \(f(0.5)=\sqrt{1.5}\).
    3. Compare your estimate with the exact value and find the absolute error.

    Work it out in the cell below. No Taylor polynomial or estimate is filled in for you.
    """)
    return


@app.cell
def _(T_3):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    x_values = np.linspace(-0.9, 2.0, 400)
    function_data = pd.DataFrame(
        {
            "x": x_values,
            "f(x)": np.sqrt(1 + x_values),
            "T_3(x)": T_3(x_values),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=function_data,
        x="x",
        y="f(x)",
        ax=ax,
        color="#4c72b0",
        linewidth=2.5,
        label="Original f(x)",
    )
    sns.lineplot(
        data=function_data,
        x="x",
        y="T_3(x)",
        ax=ax,
        color="#c44e52",
        linewidth=2.2,
        linestyle="--",
        label="Degree-3 Taylor polynomial T_3(x)",
    )
    x_axis_color = "#dd8452"
    y_axis_color = "#55a868"
    ax.axhline(
        0, color=x_axis_color, linewidth=2.4, label="x-axis (y = 0)", zorder=3
    )
    ax.axvline(
        0,
        color=y_axis_color,
        linewidth=2.4,
        label="y-axis / center (x = 0)",
        zorder=3,
    )
    ax.set_ylim(bottom=-0.2)
    ax.set_title("f(x) and its degree-3 Taylor polynomial")
    ax.set_xlabel("x", color=x_axis_color)
    ax.set_ylabel("f(x)", color=y_axis_color)
    ax.tick_params(axis="x", colors=x_axis_color)
    ax.tick_params(axis="y", colors=y_axis_color)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig
    return


@app.cell
def _():
    # Your solution goes here.
    # Find T_3(x), then use it to estimate f(0.5) and calculate the absolute error.
    return


@app.cell
def _():
    (3 / 8) * (1 + 0) ** (-5 / 2)
    return


@app.cell(hide_code=True)
def _():
    import math

    return (math,)


@app.cell
def _(math):
    def T_3(x):
        return (
            1
            + (0.5 * x)
            - (0.25 / math.factorial(2) * x**2)
            + (0.375 / math.factorial(3) * x**3)
        )

    return (T_3,)


if __name__ == "__main__":
    app.run()

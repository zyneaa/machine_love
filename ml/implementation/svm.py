import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup — `marimo`, `matplotlib`, `numpy`, `pandas`, `seaborn`

    > `import marimo as mo` · `import matplotlib.pyplot as plt` · `import numpy as np` · `import pandas as pd` · `import seaborn as sns`

    | import | role |
    |---|---|
    | `marimo as mo` | the notebook runtime — `mo.md` renders these notes, `mo.mermaid` draws the pipeline diagram |
    | `numpy as np` | the model itself: sampling, `vstack`, `concatenate`, dot products |
    | `pandas as pd` | long-form frames that seaborn can read by column name |
    | `matplotlib.pyplot as plt` | figures and axes, plus the trailing `_fig` expression that renders a cell |
    | `seaborn as sns` | theme, palettes, `scatterplot` / `barplot`, and `despine` |

    Imported once here; every later cell uses them bare. There is no other dependency — the SVM itself is built from scratch on top of numpy.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `sns.set_theme`

    > `sns.set_theme(context="notebook", style="whitegrid", palette="deep", rc=None)`

    Sets seaborn's global theme in a single call: matplotlib defaults, grid style, and default palette together.

    | argument | meaning |
    |---|---|
    | `context` | element scale — `"notebook"`, `"talk"`, `"paper"`; controls font sizes and line widths |
    | `style` | axes styling; `"whitegrid"` is a white background with a light grid |
    | `palette` | default categorical palette, here `"deep"` |
    | `rc` | dict of raw matplotlib rcParams merged on top — the escape hatch for per-element colors |

    **In this notebook:** `rc` pins one warm off-white (`#fbfbf9`) for figure, axes, and saved-image backgrounds, with muted slate text and grid lines. Every later plot inherits it, so the figures stay consistent without repeating style code.
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
    ### `np.random.default_rng`

    > `np.random.default_rng(seed=None)`

    Constructs a **Generator**, numpy's modern random source and the replacement for the legacy global `np.random.seed` state machine.

    | argument | meaning |
    |---|---|
    | `seed` | int, sequence, or `SeedSequence`. The same seed always yields the same stream — that is what makes this notebook reproducible. |

    **Returns** a `Generator` exposing `.random()`, `.normal()`, `.permutation()`, `.integers()`, and friends.

    **In this notebook:** two constants fix the problem's geometry — `n_features = 2` so the data can be plotted on a plane, and `n_samples = 500` per class so the cloud is dense enough to read. `seed=42` then fixes every draw that follows.
    """)
    return


@app.cell
def _(np):
    n_features = 2
    n_samples = 500
    rng = np.random.default_rng(seed=42)
    return n_features, n_samples, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generating dummy data for implementation
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
    | `size` | output shape — `(n_samples, n_features)` = `(500, 2)` here; `None` returns a single float |

    **In this notebook:** a first, deliberately boring dataset — 500 points in the unit square. Only the shape matters: it shows the `(rows, columns)` convention that every later array follows. The two-class data below is what the SVM actually sees.

    The trailing expressions print three views of the same object: the array, its length (`500` rows), and its `.shape` tuple (`(500, 2)`).
    """)
    return


@app.cell
def _(n_features, n_samples, rng):
    features = rng.random(size=(n_samples, n_features))
    features, len(features), features.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F2AFO

    Two samples are drawn with the same `size` but different `loc`, so only
    their centres move. The printed values confirm the sample means, and the
    histograms overlay the same draws.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Generator.normal` and `pyplot.hist`

    > `rng.normal(loc=0.0, scale=1.0, size=None)` · `plt.hist(x, bins=10, alpha=None, label=None)`

    **`Generator.normal`** samples from `N(loc, scale²)`.

    | argument | meaning |
    |---|---|
    | `loc` | the mean — where the distribution is centred |
    | `scale` | the **standard deviation**, not the variance |
    | `size` | shape of the draw, `1000` here |

    **`pyplot.hist`** bins the samples and draws the counts.

    | argument | meaning |
    |---|---|
    | `bins` | number of equal-width bins |
    | `alpha` | transparency, so overlapping bars stay readable |
    | `label` | text for the legend |

    **Note the style:** this cell uses the pyplot *state machine* (`plt.hist`, `plt.title`, `plt.show`) rather than an explicit axes object. Convenient for a quick look; later cells switch to `fig, ax = plt.subplots(...)` so layers can be styled and composed.

    **In this notebook:** only `loc` moves between the two draws (0 → 10), while the widths differ (`scale=5` then `scale=1`). The printed means land near 0 and 10, which is the point: `loc` sets the centre.
    """)
    return


@app.cell
def _(plt, rng):
    data_at_0 = rng.normal(loc=0, scale=5, size=1000)
    data_at_10 = rng.normal(loc=10, scale=1, size=1000)

    print(f"Mean when loc=0:  {data_at_0.mean():.2f}")
    print(f"Mean when loc=10: {data_at_10.mean():.2f}")

    plt.hist(data_at_0, bins=30, alpha=0.6, label="loc=0 (Mean ≈ 0)")
    plt.hist(data_at_10, bins=30, alpha=0.6, label="loc=10 (Mean ≈ 10)")

    plt.title("Effect of 'loc' on a Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Generator.normal` with a vector `loc`

    > `rng.normal(loc=[500, 2], scale=100, size=(500, 2))`

    `loc` and `scale` accept **broadcasting** shapes, so a length-2 `loc` is applied per column: column 0 is centred at 500 and column 1 at 2, both with `scale=100`.

    | argument | meaning |
    |---|---|
    | `loc` | per-column mean, stretched across all 500 rows |
    | `scale` | the per-class spread |
    | `size` | the final shape, `(500, 2)` |

    `X_pos[:11]` shows the first eleven rows: enough to read the column ordering by eye without dumping all 500. A bare `ndarray` as the last expression renders as a table.

    **In this notebook:** this is the **positive** class, centred on `(500, 2)` in feature space.
    """)
    return


@app.cell
def _(n_features, n_samples, rng):
    X_pos = rng.normal(
        loc=[n_samples, n_features], scale=100, size=(n_samples, n_features)
    )
    X_pos[:11]
    return (X_pos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Generator.normal` — the second class

    > `rng.normal(loc=[-500, -2], scale=200, size=(500, 2))`

    Same sampler, mirrored arguments: the centre flips sign and the spread **doubles**.

    | argument | meaning |
    |---|---|
    | `loc` | `[-500, -2]` — the point reflection of the positive class's centre |
    | `scale` | `200` vs. the other class's `100` |

    **In this notebook:** the wider spread is deliberate. With centres 1000 apart and spreads of 100 and 200, the clouds stay separable but not trivially so — a margin is worth having, and a few points sit closer to the boundary than the rest. That is what gives the hinge loss something to do.

    `X_neg[:11]` is the same eleven-row preview as above.
    """)
    return


@app.cell
def _(n_features, n_samples, rng):
    X_neg = rng.normal(
        loc=[-n_samples, -n_features], scale=200, size=(n_samples, n_features)
    )
    X_neg[:11]
    return (X_neg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two classes on one plot — `pd.concat`, `sns.scatterplot`, `np.ptp`

    > `pd.concat([frames], ignore_index=True)` · `sns.scatterplot(data, x, y, hue, palette, ax=...)` · `np.ptp(a, axis=0)`

    **`pd.DataFrame` + `pd.concat`** turn the two numpy arrays into a single long frame with a `class` column. `ignore_index=True` renumbers the rows `0..999` instead of repeating `0..499` twice.

    **`sns.scatterplot`** draws one marker per row, colored by `hue`.

    | argument | meaning |
    |---|---|
    | `hue` | column whose values select the color |
    | `palette` | dict mapping class names to explicit colors, so they never shift between figures |
    | `s`, `alpha`, `linewidth` | marker area, transparency, and edge width (`0` removes the outline) |
    | `ax` | the axes to draw into, which is what lets more layers share the frame |

    **`ax.scatter`** overlays each class mean as an `X`; **`np.vstack`** merges the clouds so **`np.ptp`** (max minus min) can size the padding. A detached `Line2D([], [])` adds the "class mean" entry to the legend, and **`sns.despine`** removes the top and right spines.

    **In this notebook:** the two clouds are far apart, so a straight line separates them — the setup that makes a margin meaningful, and the picture the trained hyperplane will later be drawn onto.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(X_neg, X_pos, np, pd, plt, sns):
    from matplotlib.lines import Line2D

    COLORS = {"positive": "#2563eb", "negative": "#dc2626"}
    _points = pd.concat(
        [
            pd.DataFrame(
                {
                    "x1": X_pos[:, 0],
                    "x2": X_pos[:, 1],
                    "class": "positive",
                }
            ),
            pd.DataFrame(
                {
                    "x1": X_neg[:, 0],
                    "x2": X_neg[:, 1],
                    "class": "negative",
                }
            ),
        ],
        ignore_index=True,
    )

    _fig, _ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    sns.scatterplot(
        data=_points,
        x="x1",
        y="x2",
        hue="class",
        palette=COLORS,
        s=28,
        alpha=0.55,
        linewidth=0,
        ax=_ax,
    )

    for _name, _pts in (("positive", X_pos), ("negative", X_neg)):
        _ax.scatter(
            *_pts.mean(axis=0),
            marker="X",
            s=220,
            color=COLORS[_name],
            edgecolor="white",
            linewidth=1.6,
            zorder=5,
        )

    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title(
        f"Two Gaussian classes  ·  {len(X_pos)} points each  ·  σ = 0.8",
        fontsize=13,
        pad=12,
    )

    _every = np.vstack([X_pos, X_neg])
    _pad = 0.08 * float(np.ptp(_every, axis=0).max())
    _ax.set_xlim(_every[:, 0].min() - _pad, _every[:, 0].max() + _pad)
    _ax.set_ylim(_every[:, 1].min() - _pad, _every[:, 1].max() + _pad)

    _handles, _ = _ax.get_legend_handles_labels()
    _handles.append(
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
    _ax.legend(
        handles=_handles,
        title="class",
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        borderpad=0.8,
    )

    sns.despine(ax=_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.vstack`

    > `np.vstack(tup)` — stack arrays along the first axis

    | argument | meaning |
    |---|---|
    | `tup` | sequence of arrays with matching trailing dimensions |

    `(500, 2)` and `(500, 2)` stack into `(1000, 2)`: one feature matrix, class membership dropped.

    **In this notebook:** `X` is the design matrix from here on. The label vector is built in the next cell, and the two must stay row-aligned — stack in the same order you concatenate.
    """)
    return


@app.cell
def _(X_neg, X_pos, np):
    X = np.vstack([X_pos, X_neg])
    X
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.concatenate` and the `{-1, +1}` label convention

    > `np.concatenate([a, b])` · `np.ones(n)`

    **`np.concatenate`** joins arrays along an existing axis — for two 1-D arrays of 500, the result is length 1000.

    The labels are `+1` for the positive block and `-1` for the negative block, matching the order of `np.vstack([X_pos, X_neg])`. The `±1` encoding is not cosmetic: it turns "correctly classified" into the single scalar product `t_i f(x_i) > 0`, which is what the hinge loss and the gradient both consume.

    **In this notebook:** `t[:11]` shows the first eleven labels, all `+1`; `t[100:]` starts inside the positive block and runs into the negative one, so the switch is visible.
    """)
    return


@app.cell
def _(n_samples, np):
    t = np.concatenate([np.ones(n_samples), -np.ones(n_samples)])
    t[:11], t[100:]
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Generator.permutation` and fancy indexing

    > `rng.permutation(x)` · `X[indices]`

    **`rng.permutation(len(X))`** returns a shuffled copy of the integer range `0..999` — a random ordering of the row numbers, not of the data.

    **Fancy indexing** then applies that one index array twice:

    ```python
    X_shuffled = X[indices]
    t_shuffled = t[indices]
    ```

    Same permutation, both arrays — which is the whole point. Shuffling only one of them would break the pairing between a row and its label, and the model would be training on noise.

    **In this notebook:** shuffled order matters for gradient descent, which walks the data in batches and would otherwise see 500 positives followed by 500 negatives.
    """)
    return


@app.cell
def _(X, rng, t):
    indices = rng.permutation(len(X))
    X_shuffled = X[indices]
    t_shuffled = t[indices]
    X_shuffled, t_shuffled
    return X_shuffled, t_shuffled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Boolean indexing

    > `X_shuffled[t_shuffled == 1]`

    `t_shuffled == 1` builds a length-1000 boolean mask; indexing with it keeps only the rows where the mask is `True`.

    This is the same machinery the hinge loss relies on: `margins < 1` is a mask of exactly the points that contribute to the gradient, and `X[mask]` selects them without a loop.

    **In this notebook:** the cell is a spot check — it shows the positive half of the shuffled design matrix, and confirms the labels really are aligned to rows after the permutation.
    """)
    return


@app.cell
def _(X_shuffled, t_shuffled):
    X_shuffled[t_shuffled == 1][:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decision function

    The model is a single affine map:

    $$
    f(x) = w^\top x + b
    $$

    - $w \in \mathbb{R}^{n}$ is the **normal vector** — it points perpendicular to the hyperplane $f(x) = 0$.
    - $b$ is the **bias** — changing it slides the hyperplane along $w$ without rotating it.
    - $f(x)$ is neither a probability nor a label: it is an unbounded real number.

    **Reading the number.** For a point $x$:

    $$
    f(x) = 0 \;\Rightarrow\; \text{on the boundary}, \qquad
    f(x) > 0 \;\Rightarrow\; \text{votes } +1, \qquad
    f(x) < 0 \;\Rightarrow\; \text{votes } -1
    $$

    Dividing by the norm converts the score into a signed distance:

    $$
    \frac{f(x)}{\lVert w \rVert}
    $$

    so every point on the line $f(x) = c$ is exactly $c / \lVert w \rVert$ from the boundary, and the margin lines $f(x) = \pm 1$ sit at $\pm 1/\lVert w \rVert$.

    **Why keep the raw score?** The hinge loss and its gradient are linear in $f$; thresholding here would throw away the distance information that training depends on. `predict` does the thresholding, later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `decision_function` — the model itself

    > `decision_function(X, w, b)` → `X @ w + b` · `@app.function`

    The entire model is one affine map. For a single sample:

    $$
    f(x) = w^\top x + b
    $$

    | argument | meaning |
    |---|---|
    | `X` | `(n_samples, n_features)` design matrix |
    | `w` | `(n_features,)` normal vector — the direction the hyperplane faces |
    | `b` | scalar bias — shifts the hyperplane off the origin |

    **Returns** `(n_samples,)` raw scores. Nothing is thresholded here: the sign is the predicted class, and `f(x) / ||w||` is the signed Euclidean distance from the hyperplane. Keeping scores unthresholded is what makes the hinge loss differentiable in the margin.

    `@app.function` tells marimo to keep this as an ordinary function in the module namespace — callable from any cell, with no cell-body rewriting — rather than as a cell that happens to define a name.

    **In this notebook:** because `w` starts at zero, every score starts at `b` — the flat, useless model the next few cells inspect.
    """)
    return


@app.function
def decision_function(X, w, b):
    """Return the raw score f(x) = X w + b, one value per row of X.

    The sign is the predicted class. Dividing by ||w|| turns the score into
    the signed distance from the hyperplane.
    """
    return X @ w + b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predictions

    Classification is a threshold applied to the score:

    $$
    \hat{y}(x) = \operatorname{sign}\bigl(f(x)\bigr) \in \{-1, +1\}
    $$

    with one convention to settle at the boundary itself: $f(x) = 0$ is assigned to $+1$ by testing $f(x) \ge 0$.

    **Prediction is deliberately lossy.** The whole real line collapses into two values:

    | score | label |
    |---|---|
    | $f(x) > 0$ | $+1$ |
    | $f(x) = 0$ | $+1$ (the tie-break) |
    | $f(x) < 0$ | $-1$ |

    That is all evaluation needs: accuracy, a confusion matrix, a shaded decision region. What it cannot do is drive training — the derivative of a step function is zero almost everywhere, so no gradient survives the threshold. `predict` sits at the end of the pipeline, not inside the optimisation.

    **Accuracy** is the mean agreement between prediction and truth:

    $$
    \mathrm{acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\bigl[\hat{y}(x_i) = t_i\bigr]
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `predict` — scores to labels

    > `predict(X, w, b)` → `np.where(scores >= 0, 1, -1)`

    **`np.where(cond, a, b)`** is a vectorized if/else: elementwise `a` where `cond` holds, `b` elsewhere. It returns a new array of the same shape, so the whole batch is classified in one call.

    **The tie-break is deliberate:** `scores >= 0` sends the hyperplane itself (`f(x) = 0`) to `+1`. Some convention is needed, and being explicit beats relying on `np.sign`, which answers `0` at the boundary — a value that is not a valid label.

    `predict` is the *evaluation* view of the model: the continuous score is thrown away, only the decision remains. Training uses `decision_function` directly.

    **In this notebook:** called with `w = 0`, every score is `0`, so every point is labelled `+1` — about 50% accuracy on a balanced set.
    """)
    return


@app.cell
def _(np):
    def predict(X, w, b):
        """Return +1 / -1 predictions from the sign of f(x)."""
        scores = decision_function(X, w, b)
        return np.where(scores >= 0, 1, -1)

    return (predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The hinge loss

    Every sample is scored by how far it sits on the **correct** side of the
    hyperplane, scaled by its label:

    $$
    L_i = \max\bigl(0,\; 1 - t_i f(x_i)\bigr)
    $$

    - $t_i f(x_i) \ge 1$ — correct and outside the margin: the loss is $0$.
    - $0 \le t_i f(x_i) < 1$ — correct but marginal: a linear penalty.
    - $t_i f(x_i) < 0$ — misclassified: the penalty grows without bound.

    The constant $1$ is the margin we demand. Nothing beyond it is penalised,
    which is why only points at or inside the margin — the support vectors —
    drive the solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `hinge_loss`

    > `hinge_loss(t, scores)` → `np.maximum(0, 1 - t * scores)`

    **`np.maximum(0, x)`** is the hinge: elementwise `max`, so the loss saturates at zero from below and grows linearly above.

    For each sample, `t * scores` is the margin `t_i f(x_i)`, and the loss is `max(0, 1 - margin)`:

    | margin `t f(x)` | loss | meaning |
    |---|---|---|
    | `>= 1` | `0` | correct and outside the margin — contributes nothing |
    | `0 .. 1` | `1 - margin` | correct but inside the margin — linear penalty |
    | `< 0` | `> 1` | misclassified — penalty grows without bound |

    Everything is vectorized: `t` and `scores` are length-`n` arrays, so one call returns one loss per sample. The **mean** over samples is the data term of the objective.

    **In this notebook:** this is the only place labels and predictions meet. The benchmark cell right after runs it on four hand-picked margins.
    """)
    return


@app.cell
def _(np):
    def hinge_loss(t, scores):
        """Return the per-sample hinge loss max(0, 1 - t * scores).

        `t` holds labels in {-1, +1} and `scores` the decision values f(x);
        both must have the same length.
        """
        return np.maximum(0.0, 1.0 - t * scores)

    return (hinge_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The regulariser

    $$
    R(w) = \frac{1}{2}\lVert w \rVert^2
    $$

    - Only the **weights** are penalised; the bias $b$ is deliberately left out,
      since shifting the hyperplane should not cost anything.
    - The margin width is $\dfrac{2}{\lVert w \rVert}$, so shrinking
      $\lVert w \rVert$ widens the margin — regularising *is* maximising the
      margin.
    - The $\tfrac{1}{2}$ cancels the square when differentiating
      ($\nabla_w R = w$), which keeps the gradient step clean.
    - By penalizing large weights, we encourage a wider margin.

    The full objective is $R(w) + C\sum_i L_i$, with the trade-off $C$ still to
    come. At the current $w = \mathbf{0}$ this term is exactly $0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `regularizer` — the weight penalty

    > `regularizer(w)` → `0.5 * (w @ w)` · `@app.function`

    $$
    R(w) = \tfrac{1}{2}\lVert w \rVert^2 = \tfrac{1}{2}\, w^\top w
    $$

    | argument | meaning |
    |---|---|
    | `w` | the weight vector, shape `(n_features,)` |

    **`w @ w`** is the dot product with itself, i.e. the squared norm — no `np.linalg.norm` needed, and no square root to differentiate.

    **Returns** a scalar. The bias `b` is not an argument at all, which is how "do not regularise the bias" is enforced structurally rather than by a comment.

    The `1/2` is not arbitrary: it cancels the `2` from differentiating a square, so `∇_w R = w` — the cleanest gradient in the notebook.

    **In this notebook:** since the margin width is `2 / ||w||`, shrinking `||w||` widens the margin. Regularising is maximising the margin, seen from the other side.
    """)
    return


@app.function
def regularizer(w):
    """Return R(w) = 0.5 * ||w||^2; the bias is not part of w."""
    return 0.5 * (w @ w)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The objective

    Training minimises two terms at once:

    $$
    J(w, b) = \underbrace{\frac{1}{N}\sum_{i=1}^{N}\max\bigl(0,\;1 - t_i f(x_i)\bigr)}_{\text{data term — mean hinge loss}} \;+\; \underbrace{\lambda\,\frac{1}{2}\lVert w \rVert^{2}}_{\text{model term — weight penalty}}
    $$

    - The **data term** charges nothing once every point satisfies $t_i f(x_i) \ge 1$; it grows linearly for points inside the margin or on the wrong side.
    - The **model term** charges for large weights. Because the margin is $2/\lVert w \rVert$ wide, shrinking $\lVert w \rVert$ widens it — the penalty is "maximise the margin" written as a cost.

    **The trade-off** is the whole job of $\lambda$:

    | $\lambda$ | behaviour |
    |---|---|
    | very small | the fit dominates; $\lVert w \rVert$ can grow until the margin nearly collapses |
    | very large | the margin dominates; $\lVert w \rVert$ stays small and violations are tolerated |

    **Why a mean rather than a sum?** Dividing by $N$ keeps the data term's scale independent of dataset size, so a given $\lambda$ means the same thing for 100 rows or 100 000. Textbooks usually write $C\sum_i L_i$; the two forms differ only by the constant $N$, with $C = 1/(N\lambda)$ describing the same family of problems.

    The function is **convex** in $(w, b)$ — a sum of convex hinge terms plus a quadratic — so gradient descent reaches the global minimum, not merely a local one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `objective` — data term plus model term

    > `objective(X, t, w, b, lamda_)` → `mean(hinge_loss(t, f(X))) + lamda_ * R(w)`

    $$
    J(w, b) = \underbrace{\frac{1}{N}\sum_i \max\bigl(0, 1 - t_i f(x_i)\bigr)}_{\text{mean hinge loss}} \;+\; \underbrace{\lambda\, \tfrac{1}{2}\lVert w \rVert^2}_{\text{regulariser}}
    $$

    | argument | meaning |
    |---|---|
    | `X`, `t` | design matrix and `±1` labels |
    | `w`, `b` | current parameters |
    | `lamda_` | trade-off between fitting the data and keeping `w` small; the trailing underscore avoids shadowing the Python built-in name |

    **Why a mean and not a sum:** averaging keeps the data term's scale independent of dataset size, so `lamda_` means the same thing whether there are 100 rows or 100 000. The cost is that the "sum" written in most textbooks differs from this one by a factor of `N` — the minimiser is identical.

    **In this notebook:** exactly two terms, which is why training is a two-line gradient step later. Everything up to here exists to make this one expression readable.
    """)
    return


@app.cell
def _(hinge_loss, np):
    def objective(X, t, w, b, lamda_):
        """Return the objective: mean hinge loss + lamda_ * R(w).

        Averaging the data term over samples keeps its scale independent of
        dataset size; `lamda_` trades it against the weight penalty.
        """
        scores = decision_function(X, w, b)
        return np.mean(hinge_loss(t, scores)) + regularizer(w) * lamda_

    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gradients

    The hinge is piecewise linear, so its derivative is a step:

    $$
    \frac{\partial}{\partial f}\,\max\bigl(0,\,1 - t f\bigr) =
    \begin{cases}
    0 & t f > 1 \\[2pt]
    -t & t f < 1
    \end{cases}
    $$

    Applying the chain rule to the full objective gives

    $$
    \frac{\partial J}{\partial w} = \lambda w - \frac{1}{N}\sum_{i \in V} t_i x_i,
    \qquad
    \frac{\partial J}{\partial b} = -\frac{1}{N}\sum_{i \in V} t_i
    $$

    where $V = \{\, i : t_i f(x_i) < 1 \,\}$ is the set of **margin violators**.

    **What each piece does.**

    | term | comes from | effect |
    |---|---|---|
    | $\lambda w$ | regulariser | pulls $w$ toward zero, widening the margin |
    | $-\frac{1}{N}\sum_{i\in V} t_i x_i$ | data term | averages the violating points, each pulling toward its own class |
    | $-\frac{1}{N}\sum_{i\in V} t_i$ | data term | a vote on the bias — an imbalance of violators shifts $b$ |

    **Why only the violators survive.** A point with $t_i f(x_i) > 1$ has zero loss and zero derivative: nudging the boundary cannot lower the objective for it. That is the sparsity the name *support vector* refers to, and it is why both sums run over $V$ rather than over all $N$ samples.

    **No $\lambda$ on the bias.** $b$ is deliberately outside the penalty: sliding the boundary should cost nothing, only rotating and stretching it should.

    At the kink $t_i f(x_i) = 1$ the derivative is set-valued between the two branches; any value in that interval is a valid subgradient, and the implementation takes the $t f < 1$ branch.

    **Sign convention.** These are *gradients of the objective to be minimised*, so training steps against them: $w \leftarrow w - \eta\,\partial J/\partial w$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `gradients` — subgradient of the objective

    > `gradients(X, t, w, b, lambda_)` → `(dw, db)`

    $$
    \frac{\partial J}{\partial w} = \lambda w - \frac{1}{N}\sum_{i \in V} t_i x_i
    \qquad
    \frac{\partial J}{\partial b} = -\frac{1}{N}\sum_{i \in V} t_i
    $$

    where `V` is the set of **margin violators**, `t_i f(x_i) < 1`.

    | step | what the code does |
    |---|---|
    | `margins = t * scores` | signed distance from the margin, per sample |
    | `violations = margins < 1` | the boolean mask; non-violators have zero loss, so zero gradient |
    | `X[violations].T @ t[violations]` | sums `t_i x_i` over violators only — one matmul, no loop |
    | `lambda_ * w` | the regulariser's gradient, `∇_w R = w` thanks to the `1/2` |
    | `db` | no `lambda_` term: the bias is not regularised |

    Only the violators appear, which is the computational face of the "support vectors" idea — points outside the margin push on nothing.

    **Returns** the **ascent** direction of `J`. Training therefore steps *against* it, `w -= lr * dw`.

    **In this notebook:** this is the only non-trivial derivative. Read the `db` line as a vote: each violating positive adds `-1/N`, each violating negative adds `+1/N`, and `b` moves to cancel the imbalance.
    """)
    return


@app.cell
def _(np):
    def gradients(X, t, w, b, lambda_):
        """Return (dw, db), the gradient of the objective at (w, b).

        Only margin violators (t * f(x) < 1) contribute to the loss term,
        and the bias is not regularised, so db carries no lambda_ term.
        """
        N = X.shape[0]

        scores = decision_function(X, w, b)
        margins = t * scores

        violations = margins < 1

        dw = lambda_ * w - (X[violations].T @ t[violations]) / N

        db = -np.sum(t[violations]) / N

        return dw, db

    return (gradients,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training

    Parameters move along the negative gradient, one step at a time:

    $$
    w \leftarrow w - \eta\,\frac{\partial J}{\partial w},
    \qquad
    b \leftarrow b - \eta\,\frac{\partial J}{\partial b}
    $$

    with the gradient from the previous section and a step length $\eta$ — the learning rate.

    **Reading the update.** Each step moves the hyperplane in the direction that reduces the objective fastest: the violating points pull it toward themselves, the penalty pulls $w$ back toward the origin, and $\eta$ decides how far along that compromise to travel.

    | $\eta$ | effect |
    |---|---|
    | too small | the objective barely moves; convergence needs many iterations |
    | about right | monotone decay |
    | too large | overshoot, oscillation, possible divergence |

    **Scale matters more than the number.** On raw features spanning hundreds of units the gradient is enormous, so $\eta = 1.1$ leaps far past the optimum; on standardised features the gradient is O(1) and the same $\eta$ is sensible. That is exactly why this notebook standardises before training.

    **Why one step gets its own function.** It makes the loop read as the mathematics: `train(...)` *is* the two lines above, and the training cell is just "repeat". It is also where the in-place subtlety lives — `w` is a numpy array, so `w -= ...` mutates the caller's array, while `b` is an immutable float whose new value must come back through the return value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `train` — one gradient step

    > `train(X, t, w, b, learning_rate, lambda_)` → `(w, b)`

    Wraps `gradients` and applies the descent update `w ← w − η∇_w J`,
    `b ← b − η∇_b J`.

    | argument | meaning |
    |---|---|
    | `X`, `t` | features and `±1` labels |
    | `w`, `b` | current parameters |
    | `learning_rate` | `η`, the step length |
    | `lambda_` | `λ`, the weight-penalty trade-off |

    **One caveat worth knowing before the loop below:** `w -= ...` is an *in-place*
    update on the numpy array, so the caller's `w` changes even if the return value
    is thrown away. `b` is a Python float, so `b -= ...` only rebinds the local —
    the caller's `b` never changes. That asymmetry is visible later: `w` learns,
    `b` stays at its starting value.

    **Returns** `(w, b)` for convenience.
    """)
    return


@app.cell
def _(gradients):
    def train(X, t, w, b, learning_rate, lambda_):
        """Apply one gradient-descent update and return the new parameters.

        `w` is mutated in place (numpy array); `b` is an immutable float, so
        its updated value must be taken from the return value rather than
        relying on the caller's binding.
        """
        dw, db = gradients(X, t, w, b, lambda_)

        w -= learning_rate * dw
        b -= learning_rate * db

        return w, b

    return (train,)


@app.cell
def _():
    learning_rate = 1.1
    lambda_ = 0.01
    return lambda_, learning_rate


@app.cell
def _(X_shuffled, np):
    w = np.zeros(X_shuffled.shape[1])
    b = 0.0
    return b, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardising the features

    > `mean = X.mean(axis=0)` · `std = X.std(axis=0)` · `X_scaled = (X - mean) / std`

    `axis=0` reduces over rows, so `mean` and `std` are **per column** — one number
    each for `x₁` and `x₂`. Broadcasting then centres and scales every row.

    | quantity | raw units | standardised |
    |---|---|---|
    | `x₁` centre | ≈ 0 (the classes cancel) | 0 |
    | `x₁` spread | ≈ 500 | 1 |
    | `x₂` spread | ≈ 100 | 1 |

    **Why it matters:** a weight is only meaningful relative to its feature's scale.
    On raw units `w₁ ≈ 5` is tiny and the margin `2/‖w‖` is microscopic; on
    standardised units `w ≈ 2` is large and the margin lands on the same scale as
    the data — which is what finally makes it visible in a plot.
    """)
    return


@app.cell
def _(X_shuffled):
    mean = X_shuffled.mean(axis=0)
    std = X_shuffled.std(axis=0)

    X_scaled = (X_shuffled - mean) / std
    return (X_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One step on the standardised features

    > `gradients` · `decision_function` · `objective`

    The same experiment, rerun on `X_scaled` with `learning_rate = 1.1`:

    | printout | value at the defaults |
    |---|---|
    | `dw`, `db` | the gradient at the zero model; all 1000 points violate, so the sums cover the whole set |
    | `w`, `b` | `w ≈ [1.05, 0.006]`, `b = 0` — the update runs almost purely along `x₁` |
    | `Violations` / `Satisfied` | `495` / `505` |
    | `Min` / `Max margin` | `−0.387` … `2.205` |
    | `Initial` → `Updated objective` | `1.000` → `0.121` |

    **Read the last row as the point of standardising:** `1.1` is an absurd learning
    rate by textbook standards, but on unit-variance features the gradient is O(1),
    so one step is both stable and decisive.
    """)
    return


@app.cell
def _(
    X_scaled,
    b,
    gradients,
    lambda_,
    learning_rate,
    np,
    objective,
    t_shuffled,
    w,
):
    _w = w
    _b = b

    _dw, _db = gradients(X_scaled, t_shuffled, _w, _b, lambda_)

    _w -= learning_rate * _dw
    _b -= learning_rate * _db

    _scores = decision_function(X_scaled, _w, _b)
    _margins = t_shuffled * _scores

    print("dw:", _dw)
    print("db:", _db)
    print("w:", _w)
    print("b:", _b)

    print("Violations:", np.sum(_margins < 1))
    print("Satisfied:", np.sum(_margins >= 1))
    print("Min margin:", _margins.min())
    print("Max margin:", _margins.max())

    initial_objective = objective(
        X_scaled, t_shuffled, np.zeros(X_scaled.shape[1]), 0.0, lambda_
    )

    updated_objective = objective(X_scaled, t_shuffled, _w, _b, lambda_)

    print("Initial objective:", initial_objective)
    print("Updated objective:", updated_objective)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Learning-rate sweep

    > `for lr in [0.01, 0.1, 1.0]` · `gradients` · `objective`

    One step from the zero model at three step sizes, objective printed after each:

    | `lr` | objective after one step |
    |---|---|
    | 0.01 | 0.991 |
    | 0.1 | 0.909 |
    | 1.0 | 0.161 |

    The direction `dw` is identical in all three — only the distance travelled
    changes. At `0.01` the model has barely moved; at `1.0` it has crossed most of
    the way to the optimum in a single step.

    **Note:** the loop variable is named `lr` and the cell returns it, so `lr = 1.0`
    becomes a public global afterwards.
    """)
    return


@app.cell
def _(X_scaled, gradients, lambda_, np, objective, t_shuffled):
    learning_rates = [0.01, 0.1, 1.0]

    for lr in learning_rates:
        w_test = np.zeros(X_scaled.shape[1])
        b_test = 0.0

        _dw, _db = gradients(X_scaled, t_shuffled, w_test, b_test, lambda_)

        w_test -= lr * _dw
        b_test -= lr * _db

        score = objective(X_scaled, t_shuffled, w_test, b_test, lambda_)

        print(f"LR={lr}: objective={score:.6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training loop — `epochs`, `learning rate`, `λ` are live

    > `mo.ui.slider` · `train` · `objective` · `predict`

    The same `train` step, repeated `epochs` times on `X_scaled`, recording the
    objective after every iteration.

    | widget | default | what it moves |
    |---|---|---|
    | `epochs` | 10 000 | how many gradient steps to take |
    | `learning rate` | 1.1 | step length; the sweep above shows why ~1 is plausible here |
    | `λ` | 0.01 | weight penalty — larger `λ` keeps `‖w‖` small, so `2/‖w‖` widens |

    The panel above is its own cell on purpose: marimo refuses to read a widget's
    value in the cell that created it (*"move the value access to another cell"*),
    so the sliders render in that cell and this one only reads their values.

    **Two quirks to know:**

    - The loop **continues from the current `w`** — `train` mutates the array in
      place — so lowering `epochs` after a run stops the model early instead of
      retraining it. Re-run the `w = np.zeros(...)` cell to start over.
    - `b` never moves, for the reason given under `train`: the global stays `0`, so
      the boundary stays anchored at the origin. On standardised, symmetric classes
      that is nearly correct — but it is a bug, not a feature.

    **Reading the summary:** `Misclassified` counts `t·f(x) ≤ 0`, `Inside margin`
    counts `0 < t·f(x) < 1`, `Outside margin` counts `t·f(x) ≥ 1`. At the defaults
    the run settles near **99.6% accuracy**, with `4` misclassified, `59` inside
    the margin and `937` outside — those 63 points are the ones no straight line
    satisfies.
    """)
    return


@app.cell
def _(mo):
    epochs_slider = mo.ui.slider(
        100, 20000, step=100, value=10000, label="epochs", show_value=True
    )
    lr_train = mo.ui.slider(
        steps=[0.01, 0.1, 0.5, 1.0, 1.1, 1.5, 2.0],
        value=1.1,
        label="learning rate",
        show_value=True,
    )
    lam_train = mo.ui.slider(
        0.0,
        0.2,
        step=0.005,
        value=0.01,
        label="λ (regulariser)",
        show_value=True,
    )

    mo.hstack([epochs_slider, lr_train, lam_train], justify="start", gap=1.2)
    return epochs_slider, lam_train, lr_train


@app.cell
def _(
    X_scaled,
    X_shuffled,
    b,
    epochs_slider,
    lam_train,
    lr_train,
    np,
    objective,
    predict,
    t_shuffled,
    train,
    w,
):
    epochs = int(epochs_slider.value)

    _w = w.copy()
    _b = b

    # Inspect the effect of changing b before training
    predictions = predict(X_shuffled, _w, _b)
    print("Accuracy before retraining:", np.mean(predictions == t_shuffled))

    history = []

    for epoch in range(epochs):
        _w, _b = train(
            X_scaled, t_shuffled, _w, _b, lr_train.value, lam_train.value
        )

        current_loss = objective(X_scaled, t_shuffled, _w, _b, lam_train.value)

        history.append(current_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {current_loss:.6f}")

    predictions = predict(X_scaled, _w, _b)
    accuracy = np.mean(predictions == t_shuffled)

    scores = decision_function(X_scaled, _w, _b)
    margins = t_shuffled * scores

    print("\nAccuracy:", accuracy)
    print("Misclassified:", np.sum(margins <= 0))
    print("Inside margin:", np.sum((margins > 0) & (margins < 1)))
    print("Outside margin:", np.sum(margins >= 1))

    print("\nw:", _w)
    print("b:", _b)

    weight_norm = np.linalg.norm(_w)
    print("Weight norm:", weight_norm)
    print("Geometric margin width:", 2 / weight_norm)

    w1 = _w.copy()
    b1 = _b.copy()
    return b1, history, w1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The fitted hyperplane

    > `X_scaled`, `w`, `b`, `history` from the training cell

    The same picture as the raw-feature section, but on standardised features — and
    only here does it read properly.

    - The **solid line** is the boundary `f(x) = 0`, the **dashed lines** are
      `f(x) = ±1`, and the shaded band is the margin: about **1.0** unit wide
      against a data spread of ~4, so it is finally visible.
    - The **rings** mark the points with `t·f(x) < 1` — inside the margin or on the
      wrong side. Everything unringed is satisfied and contributes nothing to the
      gradient. That is the support-vector idea, drawn.
    - The boundary passes through the origin because `b` never updates (see the
      note in the training cell). The classes are symmetric enough that this costs
      almost nothing: 4 errors out of 1000.

    `history` is in this cell's dependency list on purpose — it is the training
    cell's output, so dragging a slider in that cell re-runs this plot.
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def _(X_scaled, b1, history, np, plt, sns, t_shuffled, w1):
    _INK, _MARGIN = "#1f2937", "#0f766e"
    _COLORS = {1: "#2563eb", -1: "#dc2626"}

    _scores = X_scaled @ w1 + b1
    _margins = t_shuffled * _scores
    _viol = _margins < 1

    _norm_w = float(np.sqrt(w1 @ w1))
    _flat = _norm_w == 0.0
    _band = float("inf") if _flat else 2.0 / _norm_w
    _acc = float(((_scores >= 0) == (t_shuffled > 0)).mean())

    _fig, _ax = plt.subplots(figsize=(9.4, 7.6), constrained_layout=True)

    for _sign in (1, -1):
        _pts = X_scaled[t_shuffled == _sign]
        _ax.scatter(
            _pts[:, 0],
            _pts[:, 1],
            s=22,
            alpha=0.35,
            linewidth=0,
            color=_COLORS[_sign],
            label=f"class {_sign:+d}",
        )

    _ax.scatter(
        X_scaled[_viol, 0],
        X_scaled[_viol, 1],
        s=64,
        facecolors="none",
        edgecolors="#111827",
        linewidths=0.8,
        zorder=5,
        label=f"inside margin ({int(_viol.sum())})",
    )

    _lim = 1.08 * float(np.abs(X_scaled).max())
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal", adjustable="box")

    if _flat:
        _ax.text(
            0.5,
            0.5,
            "w = 0 — run the training cell first",
            transform=_ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
        )
    else:

        def _iso(c):
            "Endpoints of {x : w·x + b = c} across the visible box."
            if abs(w1[0]) >= abs(w1[1]):
                _yy = np.array([-_lim, _lim])
                return (c - b1 - w1[1] * _yy) / w1[0], _yy
            _xx = np.array([-_lim, _lim])
            return _xx, (c - b1 - w1[0] * _xx) / w1[1]

        _yy = np.array([-_lim, _lim])

        _ax.fill_betweenx(
            _yy,
            (-1.0 - b1 - w1[1] * _yy) / w1[0],
            (1.0 - b1 - w1[1] * _yy) / w1[0],
            color=_MARGIN,
            alpha=0.16,
            lw=0,
            zorder=2,
        )

        for _c, _color, _lw, _style, _label in (
            (0.0, _INK, 2.2, "-", "boundary  f(x) = 0"),
            (1.0, _MARGIN, 1.4, (0, (4, 3)), "margin  f(x) = ±1"),
            (-1.0, _MARGIN, 1.4, (0, (4, 3)), None),
        ):
            _xs, _ys = _iso(_c)
            _ax.plot(
                _xs,
                _ys,
                color=_color,
                lw=_lw,
                ls=_style,
                label=_label,
                zorder=4,
            )

    _ax.set_xlabel(r"$x_1$  (standardised)")
    _ax.set_ylabel(r"$x_2$  (standardised)")
    _ax.set_title(
        f"SVM after {len(history):,} epochs   ·   accuracy {_acc:.2%}   ·   "
        + ("w = 0" if _flat else f"margin 2/||w|| = {_band:.2f}"),
        fontsize=12.5,
        pad=12,
    )

    _ax.text(
        0.02,
        0.02,
        f"misclassified {int((_margins <= 0).sum())}   ·   "
        f"inside margin {int(((_margins > 0) & (_margins < 1)).sum())}   ·   "
        f"outside {int((_margins >= 1).sum())}\n"
        f"w = [{w1[0]:.3f}, {w1[1]:.3f}]   ·   b = {b1:.4f}   ·   "
        f"||w|| = {_norm_w:.3f}",
        transform=_ax.transAxes,
        fontsize=9,
        color="#475569",
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#fbfbf9",
            edgecolor="#e2e8f0",
            alpha=0.92,
        ),
        zorder=7,
    )

    _ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92)
    sns.despine(ax=_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training progress

    > `history` · `plt.plot`

    One objective value per epoch, from the zero model down to the settled value.

    | shape | diagnosis |
    |---|---|
    | smooth monotone decay | the step size is below the stability limit — the healthy case |
    | decay with visible wiggles | points crossing in and out of the margin; normal for hinge loss |
    | flat from the first epoch | rate too small, or `λ` so large it pins `‖w‖` |
    | rising or exploding | rate above the stability limit — step the value down |

    **Two things to note when reading it:** the curve is linear in the objective, so
    most of the action is in the first few hundred epochs and the tail flattens;
    `plt.yscale("log")` would make that tail legible. And because the loop resumes
    from the current `w`, re-running with more epochs continues this curve from
    where it stopped rather than redrawing it.
    """)
    return


@app.cell
def _(history, plt):
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("SVM Training Progress")
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive: the main knobs

    > `w₁ start`, `w₂ start`, `b start` · `learning rate`, `λ`, `steps`

    Every control feeds one short gradient-descent run, and everything
    downstream — the figure, the readout, the history table — recomputes when a
    slider moves.

    | control | what it changes |
    |---|---|
    | `w₁ start`, `w₂ start` | where the weight vector begins; `[0, 0]` is the flat model |
    | `b start` | the starting offset — shifts the hyperplane without rotating it |
    | `learning rate` | step length along the negative gradient |
    | `λ` | the weight penalty: larger `λ` keeps `‖w‖` small, so the margin `2/‖w‖` widens |
    | `steps` | number of gradient steps; `0` leaves the start untouched |

    With the defaults (`w = [0, 0]`, `b = 0`, `0.01`, `λ = 0.01`, `2` steps) the
    run starts from the flat model and lands on the three violators tabulated
    below.

    ### What the figure shows

    One step already leaves **3 of 1000** instances with `t·f(x) < 1`, all of
    them class `-1` points sitting on the positive side of the boundary:

    | $x_1$ | $x_2$ | $t$ | $f(x)$ | $t\,f(x)$ |
    |---|---|---|---|---|
    | 71.5 | 58.2 | $-1$ | 363.1 | $-363.1$ |
    | 104.8 | $-145.2$ | $-1$ | 530.1 | $-530.1$ |
    | 190.8 | 55.0 | $-1$ | 968.2 | $-968.2$ |

    The solid line is the boundary $f(x) = 0$; the dashed lines are the margin
    lines $f(x) = \pm 1$ with the band between them shaded; the black rings are
    the violators, drawn over their own class colour. The left panel is
    everything, the dashed rectangle marks the zoom, and the right panel shows
    that zoom with each violator annotated by its margin.

    **A scale warning:** at the defaults $\lVert w \rVert = 5.07$, so the margin
    is $2/\lVert w \rVert = 0.39$ units against a data span of about
    $1{,}900$. The two dashed lines therefore sit on top of the boundary and the
    band is invisible — that is the honest picture, not a plotting bug. Raising
    `learning rate` or lowering `λ` grows $\lVert w \rVert$ and *narrows* the
    margin further; nothing on these sliders makes the band visible at this data
    scale, which is itself the lesson: `2/‖w‖` only compares to data distances
    once the features are standardised.
    """)
    return


@app.cell
def _(mo):
    w1_start = mo.ui.slider(
        -10.0, 10.0, step=0.5, value=0.0, label="w₁ start", show_value=True
    )
    w2_start = mo.ui.slider(
        -10.0, 10.0, step=0.5, value=0.0, label="w₂ start", show_value=True
    )
    b_start = mo.ui.slider(
        -500.0, 500.0, step=10.0, value=0.0, label="b start", show_value=True
    )
    lr_slider = mo.ui.slider(
        steps=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.01,
        label="learning rate",
        show_value=True,
    )
    lam = mo.ui.slider(
        0.0,
        0.2,
        step=0.005,
        value=0.01,
        label="λ (regulariser)",
        show_value=True,
    )
    n_steps = mo.ui.slider(
        0, 200, step=1, value=2, label="steps", show_value=True
    )

    mo.vstack(
        [
            mo.hstack([w1_start, w2_start, b_start], justify="start", gap=1.2),
            mo.hstack([lr_slider, lam, n_steps], justify="start", gap=1.2),
        ],
        gap=0.8,
    )
    return b_start, lam, lr_slider, n_steps, w1_start, w2_start


@app.cell
def _(
    X_shuffled,
    b_start,
    lam,
    lr_slider,
    n_steps,
    np,
    objective,
    t_shuffled,
    train,
    w1_start,
    w2_start,
):
    _w = np.array([float(w1_start.value), float(w2_start.value)])
    _b = float(b_start.value)
    _lr, _lam = float(lr_slider.value), float(lam.value)
    _steps = int(n_steps.value)

    fit_history = []
    for _step in range(_steps + 1):
        _scores = X_shuffled @ _w + _b
        _margins = t_shuffled * _scores
        fit_history.append(
            {
                "step": _step,
                "w1": round(float(_w[0]), 4),
                "w2": round(float(_w[1]), 5),
                "b": round(float(_b), 4),
                "objective": round(
                    float(objective(X_shuffled, t_shuffled, _w, _b, _lam)), 4
                ),
                "violations": int((_margins < 1).sum()),
                "accuracy %": round(
                    100.0 * float(((_scores >= 0) == (t_shuffled > 0)).mean()),
                    2,
                ),
            }
        )
        if _step == _steps:
            break
        _w, _b = train(X_shuffled, t_shuffled, _w, _b, _lr, _lam)

    w_fit, b_fit = _w, _b
    return b_fit, fit_history, w_fit


@app.cell
def _(fit_history, mo, np, w_fit):
    _last = fit_history[-1]
    _norm = float(np.sqrt(w_fit @ w_fit))
    _band = "∞" if _norm == 0.0 else f"{2.0 / _norm:.3f}"

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{_last['violations']}",
                        label="violations",
                        caption="t·f < 1",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{_last['objective']:.3f}",
                        label="objective J",
                        caption="at the final step",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{_last['accuracy %']:.1f}%",
                        label="accuracy",
                        caption="on the shuffled set",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{_norm:.3f}",
                        label="‖w‖",
                        caption="margin 2/‖w‖ = " + _band,
                        bordered=True,
                    ),
                ],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
            mo.ui.table(
                fit_history,
                selection=None,
                pagination=True,
                page_size=10,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
            ),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(X_shuffled, b_fit, np, plt, sns, t_shuffled, w_fit):
    from matplotlib.patches import Rectangle

    _INK, _MARGIN = "#1f2937", "#0f766e"
    _COLORS = {1: "#2563eb", -1: "#dc2626"}
    _NAMES = {1: "class +1", -1: "class −1"}

    _scores = X_shuffled @ w_fit + b_fit
    _margins = t_shuffled * _scores
    _viol = _margins < 1
    _norm_w = float(np.sqrt(w_fit @ w_fit))
    _flat = _norm_w == 0.0
    _band = float("inf") if _flat else 2.0 / _norm_w
    _few = 0 < int(_viol.sum()) <= 25

    def _limits(pts, pad=0.06):
        "Window around the data, padded by a fraction of its own span."
        _pad_x = pad * float(np.ptp(pts[:, 0]))
        _pad_y = pad * float(np.ptp(pts[:, 1]))
        return (
            (pts[:, 0].min() - _pad_x, pts[:, 0].max() + _pad_x),
            (pts[:, 1].min() - _pad_y, pts[:, 1].max() + _pad_y),
        )

    def _iso(offset, xlim, ylim):
        "Endpoints of {x : w·x + b = offset} across the visible window."
        if abs(w_fit[0]) >= abs(w_fit[1]):
            _ys = np.array(ylim, dtype=float)
            return (offset - b_fit - w_fit[1] * _ys) / w_fit[0], _ys
        _xs = np.array(xlim, dtype=float)
        return _xs, (offset - b_fit - w_fit[0] * _xs) / w_fit[1]

    def _panel(ax, xlim, ylim, annotate):
        for _sign in (1, -1):
            _pts = X_shuffled[t_shuffled == _sign]
            ax.scatter(
                _pts[:, 0],
                _pts[:, 1],
                s=15,
                alpha=0.35,
                linewidth=0,
                color=_COLORS[_sign],
                label=_NAMES[_sign],
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if _flat:
            ax.text(
                0.5,
                0.5,
                "w = 0 — no direction, no hyperplane",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#64748b",
            )
        else:
            # the margin band: -1 <= f(x) <= 1
            _ys_fill = np.linspace(*ylim, 256)
            ax.fill_betweenx(
                _ys_fill,
                (-1.0 - b_fit - w_fit[1] * _ys_fill) / w_fit[0],
                (1.0 - b_fit - w_fit[1] * _ys_fill) / w_fit[0],
                color=_MARGIN,
                alpha=0.20,
                lw=0,
                zorder=1,
            )
            for _offset, _color, _lw, _style, _label in (
                (0.0, _INK, 2.0, "-", "boundary  f(x) = 0"),
                (1.0, _MARGIN, 1.3, (0, (4, 3)), "margin  f(x) = ±1"),
                (-1.0, _MARGIN, 1.3, (0, (4, 3)), None),
            ):
                _xs, _ys = _iso(_offset, xlim, ylim)
                ax.plot(
                    _xs,
                    _ys,
                    color=_color,
                    lw=_lw,
                    ls=_style,
                    label=_label,
                    zorder=3,
                )

        # the violators, ringed so they stand out from their own class
        ax.scatter(
            X_shuffled[_viol, 0],
            X_shuffled[_viol, 1],
            s=190 if _few else 70,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.7 if _few else 0.8,
            zorder=5,
            label="violating instance  (t·f < 1)",
        )

        if annotate:
            for _i in np.flatnonzero(_viol):
                ax.annotate(
                    f"t·f = {_margins[_i]:,.0f}",
                    (X_shuffled[_i, 0], X_shuffled[_i, 1]),
                    xytext=(14, 9),
                    textcoords="offset points",
                    fontsize=9.5,
                    color="#111827",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="#fbfbf9",
                        edgecolor="none",
                        alpha=0.92,
                    ),
                    zorder=6,
                )

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend(
            loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92
        )
        sns.despine(ax=ax)

    _fig, (_ax_all, _ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(15.2, 6.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    _xlim, _ylim = _limits(X_shuffled)
    _panel(_ax_all, _xlim, _ylim, annotate=False)
    _ax_all.set_title(
        "All instances, after the selected steps", fontsize=11.5, pad=12
    )

    if _few:
        _vpts = X_shuffled[_viol]
        _zxl = (
            min(float(_vpts[:, 0].min()), 0.0) - 70.0,
            float(_vpts[:, 0].max()) + 70.0,
        )
        _zyl = (
            float(_vpts[:, 1].min()) - 70.0,
            float(_vpts[:, 1].max()) + 70.0,
        )
    else:
        _zxl, _zyl = _xlim, _ylim

    _panel(_ax_zoom, _zxl, _zyl, annotate=_few)
    _ax_zoom.set_title(
        f"Zoom: the {int(_viol.sum())} violating instances"
        if _few
        else f"{int(_viol.sum())} instances violate the margin",
        fontsize=11.5,
        pad=12,
    )

    _ax_all.add_patch(
        Rectangle(
            (_zxl[0], _zyl[0]),
            _zxl[1] - _zxl[0],
            _zyl[1] - _zyl[0],
            fill=False,
            edgecolor="#64748b",
            lw=1.1,
            ls=(0, (4, 3)),
            zorder=4,
        )
    )

    _band_text = "undefined (w = 0)" if _flat else f"{_band:.2f} units"
    _ax_all.text(
        0.02,
        0.02,
        f"margin band 2/||w|| = {_band_text}\n"
        f"data spans {np.ptp(X_shuffled[:, 0]):,.0f}, so f = ±1 overlap the boundary",
        transform=_ax_all.transAxes,
        fontsize=8.5,
        color="#475569",
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#fbfbf9",
            edgecolor="#e2e8f0",
            alpha=0.92,
        ),
        zorder=7,
    )

    _fig.suptitle(
        f"w = [{w_fit[0]:.3f}, {w_fit[1]:.4f}],  b = {b_fit:.1e},  "
        f"||w|| = {_norm_w:.3f}   ·   "
        f"{int(_viol.sum())} of {len(X_shuffled)} instances violate the margin",
        fontsize=13.5,
        fontweight="bold",
        color=_INK,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialise the parameters

    Training starts from zero: $w = \mathbf{0}$ and $b = 0$, so every score is
    $0$ and `predict` returns $+1$ everywhere. These are temporary private names
    (`_w`, `_b`); they become the shared public `w` and `b` once the training
    step lands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the loss looks like, and what the model does right now

    **Left:** the hinge as a function of the margin $m = t\,f(x)$. It is exactly
    zero once $m \ge 1$, ramps down linearly inside the margin, and grows
    without bound for negative margins. The red dots are the four example
    scores above, sitting at their own losses.

    **Right:** the current state of the model. With $w = 0$ and $b = 0$ every
    score is $0$, so `predict` answers $+1$ for every sample — right half the
    time, by luck. There is no hyperplane to draw yet, because a zero normal
    vector does not define a direction; a real $w$ from training is what will
    put a line on the scatter plot above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two panels — `np.linspace`, `axvspan`, `DataFrame.melt`, `sns.barplot`

    > `np.linspace(a, b, n)` · `ax.axvspan(x1, x2, alpha=...)` · `df.melt(id_vars, var_name, value_name)` · `sns.barplot(data, x, y, hue, ...)`

    | call | role here |
    |---|---|
    | `np.linspace(-3, 4, 500)` | a dense margin axis, so the drawn hinge is a smooth curve rather than a polyline |
    | `ax.axvspan` | shades the three regimes — misclassified, inside margin, no penalty — behind the curve |
    | `ax.axvline` / `ax.axhline` | marks the `margin = 1` threshold and the zero baseline |
    | `ax.annotate`, `ax.text` | labels the four example points from the earlier table |
    | `df.melt` | turns the wide `actual` / `predicted` counts into long form, so seaborn can group them by one `hue` column |
    | `sns.barplot` | draws the paired counts; `palette` keeps actual grey and predicted blue |
    | `sns.despine` | drops the top and right spines on both panels |

    `fig, (ax1, ax2) = plt.subplots(1, 2, ...)` allocates two axes at once; `constrained_layout=True` handles spacing so no `tight_layout()` call is needed.

    **In this notebook:** the left panel is the loss *definition* drawn to scale, the right panel is the model's current behaviour — every prediction `+1`, accuracy `50%`. The pair makes the gap explicit: the loss is already implemented, the fit is the only thing missing.
    """)
    return


@app.cell(hide_code=True)
def _(
    X_shuffled,
    hinge_loss,
    n_features,
    np,
    pd,
    plt,
    predict,
    sns,
    t_shuffled,
):
    _fig, (_ax_loss, _ax_pred) = plt.subplots(
        1, 2, figsize=(14.5, 5.4), constrained_layout=True
    )

    _m = np.linspace(-3.0, 4.0, 500)
    _l = hinge_loss(1.0, _m)

    _ax_loss.axvspan(-3.0, 0.0, color="#dc2626", alpha=0.07, lw=0)
    _ax_loss.axvspan(0.0, 1.0, color="#d97706", alpha=0.09, lw=0)
    _ax_loss.axvspan(1.0, 4.0, color="#0f766e", alpha=0.07, lw=0)

    _ax_loss.plot(_m, _l, color="#1f2937", lw=2.4, label=r"$L=\max(0,\,1-m)$")
    _ax_loss.axvline(1.0, color="#0f766e", ls=":", lw=1.5)
    _ax_loss.axhline(0.0, color="#cbd5e1", lw=1.0, zorder=0)

    _example_margins = np.array([3.0, 0.5, -2.0, 1.0])
    _example_losses = hinge_loss(1.0, _example_margins)
    _ax_loss.scatter(
        _example_margins,
        _example_losses,
        s=70,
        color="#e11d48",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="example margins",
    )
    for _mx, _lx in zip(_example_margins, _example_losses):
        _ax_loss.annotate(
            f"{_lx:g}",
            (_mx, _lx),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            color="#e11d48",
        )

    _ax_loss.text(
        -2.95, 0.22, "misclassified", color="#dc2626", fontsize=9, va="bottom"
    )
    _ax_loss.text(
        0.5,
        2.1,
        "inside margin",
        color="#d97706",
        fontsize=9,
        rotation=90,
        ha="center",
        va="center",
    )
    _ax_loss.text(
        1.55, 0.22, "no penalty", color="#0f766e", fontsize=9, va="bottom"
    )

    _ax_loss.set_xlim(-3.0, 4.0)
    _ax_loss.set_ylim(-0.15, 4.2)
    _ax_loss.set_xlabel(r"margin  $t_i\,f(x_i)$")
    _ax_loss.set_ylabel(r"loss  $L_i$")
    _ax_loss.set_title(
        "Hinge loss: flat once the margin is met", fontsize=11.5, pad=12
    )
    _ax_loss.legend(loc="upper right", frameon=True, framealpha=0.92)
    sns.despine(ax=_ax_loss)

    _w = np.zeros(n_features)
    _b = 0.0
    _predictions = predict(X_shuffled, _w, _b)
    _accuracy = float((_predictions == t_shuffled).mean())

    _counts = pd.DataFrame(
        {
            "class": ["-1", "+1"],
            "actual": [
                int((t_shuffled == -1).sum()),
                int((t_shuffled == 1).sum()),
            ],
            "predicted": [
                int((_predictions == -1).sum()),
                int((_predictions == 1).sum()),
            ],
        }
    ).melt(id_vars="class", var_name="source", value_name="count")

    sns.barplot(
        data=_counts,
        x="class",
        y="count",
        hue="source",
        palette={"actual": "#94a3b8", "predicted": "#2563eb"},
        width=0.62,
        saturation=0.92,
        edgecolor="#fbfbf9",
        linewidth=1.5,
        ax=_ax_pred,
    )
    for _bar, _count in zip(_ax_pred.patches, _counts["count"]):
        _ax_pred.annotate(
            f"{int(_count)}",
            (_bar.get_x() + _bar.get_width() / 2, _count),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color="#1f2937",
        )

    _ax_pred.set_ylim(0.0, float(_counts["count"].max()) * 1.2)
    _ax_pred.set_xlabel("class  $t$")
    _ax_pred.set_ylabel("samples")
    _ax_pred.set_title(
        f"Untrained parameters: every point predicted +1  ·  accuracy {_accuracy:.0%}",
        fontsize=11.5,
        pad=12,
    )
    _ax_pred.legend(
        title="", loc="upper left", frameon=True, framealpha=0.92, fontsize=9
    )
    sns.despine(ax=_ax_pred)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the pieces connect

    Everything defined so far is a small pure function, and they compose in one
    direction:

    1. `decision_function(X, w, b)` — the only place features and parameters meet.
    2. `predict(X, w, b)` — collapses scores into labels $\pm 1$.
    3. `hinge_loss(t, scores)` — turns labels plus scores into one loss per sample.
    4. `regularizer(w)` — scores the weights alone, ignoring `b`.

    Put together, they are the two halves of the SVM objective: the **data term**
    $\sum_i L_i$ and the **model term** $R(w)$. The dashed nodes are the one
    remaining step — combine them with the trade-off $C$ and minimise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    flowchart LR
        X["X<br/>n_samples x n_features"] --> DF
        W["w"] --> DF
        B["b"] --> DF

        DF["decision_function(X, w, b)"] --> S["scores<br/>f(x) = Xw + b"]
        S --> P["predict(X, w, b)"]
        S --> H["hinge_loss(t, scores)"]
        T["t = -1 or +1"] --> H

        P --> Y["labels<br/>sign f(x)"]
        H --> L["losses<br/>L_i = max(0, 1 - t_i f_i)"]

        W --> R["regularizer(w)"]
        R --> RV["R(w) = 0.5 * ||w||^2"]

        L --> O["objective (next)<br/>R(w) + C * sum L_i"]
        RV --> O
        O -.-> FIT["minimise over w, b"]

        classDef input fill:#e2e8f0,stroke:#64748b,color:#1f2937;
        classDef fn fill:#ccfbf1,stroke:#0f766e,color:#0f172a;
        classDef val fill:#fef3c7,stroke:#d97706,color:#1f2937;
        classDef todo fill:#ffffff,stroke:#94a3b8,stroke-dasharray:5 5,color:#475569;
        class X,W,B,T input;
        class DF,P,H,R fn;
        class S,Y,L,RV val;
        class O,FIT todo;
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sanity checks with dummy data

    Each check below is independent, deterministic, and uses a tiny local dataset. These checks validate the model functions; the training workflow above remains unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test: decision scores and predictions

    Check the affine score and the `f(x) = 0` tie-break using three hand-built points.
    """)
    return


@app.cell
def _(mo, np, predict):
    _test_X = np.array([[2.0, 1.0], [-1.0, 3.0], [0.0, 0.5]])
    _test_w = np.array([1.0, -1.0])
    _test_b = 0.5
    _test_scores = decision_function(_test_X, _test_w, _test_b)
    _test_predictions = predict(_test_X, _test_w, _test_b)

    assert np.allclose(_test_scores, [1.5, -3.5, 0.0])
    assert np.array_equal(_test_predictions, [1, -1, 1])
    mo.md(
        "Decision scores and predictions pass, including the zero-score tie-break."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Test: hinge loss

    Four fixed score/label pairs cover zero loss, an inside-margin point, and a misclassified point. Expected losses: `[0, 0.5, 0, 2]`.
    """)
    return


@app.cell
def _(hinge_loss, mo, np):
    _test_scores = np.array([3.0, 0.5, -2.0, 1.0])
    _test_labels = np.array([1.0, 1.0, -1.0, -1.0])
    _test_losses = hinge_loss(_test_labels, _test_scores)

    assert np.allclose(_test_losses, [0.0, 0.5, 0.0, 2.0])
    assert np.isclose(_test_losses.mean(), 0.625)
    mo.md("Hinge-loss values and mean pass.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Test: regularizer

    Verify `R([3, 4]) = 12.5` and doubling weights quadruples the penalty.
    """)
    return


@app.cell
def _(mo, np):
    _test_w = np.array([3.0, 4.0])
    _test_penalty = regularizer(_test_w)

    assert np.isclose(_test_penalty, 12.5)
    assert np.isclose(regularizer(2.0 * _test_w), 4.0 * _test_penalty)
    mo.md("Regularizer value and quadratic scaling pass.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Test: objective

    Use two opposite, correctly labelled points with fixed weights. Their margins are both `0.5`, so mean hinge loss is `0.5`; regularization adds `0.025`.
    """)
    return


@app.cell
def _(mo, np, objective):
    _test_X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _test_t = np.array([1.0, -1.0])
    _test_w = np.array([0.5, 0.0])
    _test_b = 0.0
    _test_lambda = 0.2
    _test_objective = objective(
        _test_X, _test_t, _test_w, _test_b, _test_lambda
    )

    assert np.isclose(_test_objective, 0.525)
    mo.md(f"Objective matches expected value: `{_test_objective:.3f}`.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Test: analytical gradients vs. finite differences

    This fixed dataset keeps all margins away from the hinge kink, so centered finite differences can verify both gradients.
    """)
    return


@app.cell
def _(np, objective):
    def numerical_gradient_w(X, t, w, b, lambda_, eps=1e-5):
        grad = np.zeros_like(w)

        for j in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()

            w_plus[j] += eps
            w_minus[j] -= eps

            loss_plus = objective(X, t, w_plus, b, lambda_)
            loss_minus = objective(X, t, w_minus, b, lambda_)

            grad[j] = (loss_plus - loss_minus) / (2 * eps)

        return grad

    return (numerical_gradient_w,)


@app.cell
def _(objective):
    def numerical_gradient_b(X, t, w, b, lambda_, eps=1e-5):
        loss_plus = objective(X, t, w, b + eps, lambda_)
        loss_minus = objective(X, t, w, b - eps, lambda_)

        return (loss_plus - loss_minus) / (2 * eps)

    return (numerical_gradient_b,)


@app.cell
def _(gradients, mo, np, numerical_gradient_b, numerical_gradient_w):
    # Compare the analytical gradient with centered finite differences.
    _test_X = np.array([[1.0, 0.5], [-1.0, 0.25], [0.0, -1.0]])
    _test_t = np.array([1.0, -1.0, 1.0])
    _test_w = np.array([0.2, -0.3])
    _test_b = 0.1
    _test_lambda = 0.1

    _test_analytical_dw, _test_analytical_db = gradients(
        _test_X, _test_t, _test_w, _test_b, _test_lambda
    )
    _test_numerical_dw = numerical_gradient_w(
        _test_X, _test_t, _test_w, _test_b, _test_lambda
    )
    _test_numerical_db = numerical_gradient_b(
        _test_X, _test_t, _test_w, _test_b, _test_lambda
    )

    assert np.allclose(_test_analytical_dw, _test_numerical_dw, atol=1e-6)
    assert np.isclose(_test_analytical_db, _test_numerical_db, atol=1e-6)
    print("dw:", _test_analytical_dw, "≈", _test_numerical_dw)
    print("db:", _test_analytical_db, "≈", _test_numerical_db)
    mo.md("Analytical and numerical gradients agree.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Test: one training step

    A tiny separable dataset checks that `train` updates the weights as expected and lowers the objective.
    """)
    return


@app.cell
def _(mo, np, objective, train):
    _test_X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _test_t = np.array([1.0, -1.0])
    _test_w = np.zeros(2)
    _test_b = 0.0
    _test_lambda = 0.1
    _test_initial_objective = objective(
        _test_X, _test_t, _test_w, _test_b, _test_lambda
    )

    _test_w, _test_b = train(
        _test_X,
        _test_t,
        _test_w,
        _test_b,
        learning_rate=0.1,
        lambda_=_test_lambda,
    )
    _test_updated_objective = objective(
        _test_X, _test_t, _test_w, _test_b, _test_lambda
    )

    assert np.allclose(_test_w, [0.1, 0.0])
    assert np.isclose(_test_b, 0.0)
    assert _test_updated_objective < _test_initial_objective
    mo.md(
        f"One step gives `w={_test_w.tolist()}`, `b={_test_b:.1f}` and "
        f"reduces objective `{_test_initial_objective:.3f} → {_test_updated_objective:.4f}`."
    )
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys

    assert sys.version_info >= (3, 7)
    return


@app.cell
def _():
    from packaging import version
    import sklearn

    assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
    print(sklearn.__version__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Downloading dataset
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import kagglehub
    import numpy as np

    dataset_name = "algozee/teenager-menthal-healy"

    download_path = kagglehub.dataset_download(dataset_name)
    csv_file = list(Path(download_path).glob("*.csv"))[0]
    data = pd.read_csv(csv_file)
    return Path, data, np, pd


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data["daily_social_media_hours"].value_counts(normalize=True)
    return


@app.cell
def _(Path):
    import matplotlib.pyplot as plt 

    IMAGES_PATH = Path() / "images" / "social_media_impact"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    return (plt,)


@app.cell
def _(data, plt):
    plt.rc("font", size=10)
    plt.rc("axes", labelsize=10, titlesize=14)
    plt.rc("legend", fontsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)

    data.hist(bins=50, figsize=(14, 14))
    # save_fig("social media data")
    plt.show()
    return


@app.cell
def _(data):
    data["depression_label"].value_counts()
    return


@app.cell
def _(data):
    from sklearn.model_selection import train_test_split

    strat_train_set, strat_test_set = train_test_split(
        data, test_size=0.2, stratify=data["social_interaction_level"], random_state=6112001)

    print(strat_train_set["social_interaction_level"].value_counts(normalize=True))
    print(data["social_interaction_level"].value_counts(normalize=True))
    return (strat_train_set,)


@app.cell
def _(strat_train_set):
    temp_df = strat_train_set.copy()

    temp_df["interaction"] = temp_df["anxiety_level"] * temp_df["daily_social_media_hours"]
    temp_df["gap"] = temp_df["stress_level"] - temp_df["sleep_hours"]
    temp_df["success"] = temp_df["age"] * temp_df["academic_performance"]
    return (temp_df,)


@app.cell
def _(temp_df):
    corr_matrix = temp_df.corr(numeric_only=True)
    corr_matrix["addiction_level"].sort_values(ascending=False)
    return


@app.cell
def _(plt, temp_df):
    plt.rc("font", size=10)
    plt.rc("axes", labelsize=10, titlesize=14)
    plt.rc("legend", fontsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)

    temp_df.hist(bins=50, figsize=(14, 14))
    plt.show()
    return


@app.cell
def _(strat_train_set):
    addiction = strat_train_set.drop("addiction_level", axis=1)
    addiction_labels = strat_train_set["addiction_level"].copy()
    return addiction, addiction_labels


@app.cell
def _(addiction):
    addiction.head()
    return


@app.cell
def _(addiction_labels):
    addiction_labels.head()
    return


@app.cell
def _(addiction):
    from sklearn.preprocessing import OrdinalEncoder

    order = ["low", "medium", "high"]
    ordinal_encoder = OrdinalEncoder(categories=[order])

    interaction_cat_ord = ordinal_encoder.fit_transform(addiction[["social_interaction_level"]])

    interaction_cat_ord[:20]
    return OrdinalEncoder, ordinal_encoder


@app.cell
def _(ordinal_encoder):
    ordinal_encoder.categories_
    return


@app.cell
def _(addiction):
    gender_cat = addiction[["gender"]]
    platform_cat = addiction[["platform_usage"]]
    return gender_cat, platform_cat


@app.cell
def _(gender_cat, platform_cat):
    print(gender_cat, platform_cat)
    return


@app.cell
def _(gender_cat, platform_cat):
    from sklearn.preprocessing import OneHotEncoder

    gender_1hot_encoder = OneHotEncoder()
    platform_1hot_encoder = OneHotEncoder()

    gender_cat_1hot = gender_1hot_encoder.fit_transform(gender_cat)
    platform_cat_1hot = platform_1hot_encoder.fit_transform(platform_cat)
    return (
        OneHotEncoder,
        gender_1hot_encoder,
        gender_cat_1hot,
        platform_1hot_encoder,
        platform_cat_1hot,
    )


@app.cell
def _(gender_cat_1hot):
    gender_cat_1hot.toarray()
    return


@app.cell
def _(platform_cat_1hot):
    platform_cat_1hot.toarray()
    return


@app.cell
def _(gender_1hot_encoder, platform_1hot_encoder):
    gender_1hot_encoder.categories_, platform_1hot_encoder.categories_, gender_1hot_encoder.get_feature_names_out(), platform_1hot_encoder.get_feature_names_out()
    return


@app.cell
def _():
    from sklearn import set_config

    set_config(display="diagram")
    return


@app.cell
def _(OneHotEncoder, OrdinalEncoder, addiction, addiction_labels, np):
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import make_pipeline

    X1_train = addiction.copy()
    y1_train = addiction_labels.copy()

    ordinal_cols = ["social_interaction_level"]
    onehot_cols = ["gender", "platform_usage"]

    num_attribs = X1_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = X1_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessing = ColumnTransformer([
        ("ord", OrdinalEncoder(categories=[["low", "medium", "high"]]), ordinal_cols),
        ("1hot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
    ], remainder="passthrough")

    preprocessing
    return LinearRegression, make_pipeline, preprocessing


@app.cell
def _(addiction, preprocessing):
    housing_prepared = preprocessing.fit_transform(addiction)
    return


@app.cell
def _(preprocessing):
    preprocessing.get_feature_names_out()
    return


@app.cell
def _(
    LinearRegression,
    addiction,
    addiction_labels,
    make_pipeline,
    preprocessing,
):
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(addiction, addiction_labels)
    return (lin_reg,)


@app.cell
def _(addiction, addiction_labels, addiction_predictions, lin_reg):
    addiction_predictions_lr = lin_reg.predict(addiction)
    addiction_predictions[:5], addiction_labels.iloc[:5].values
    return (addiction_predictions_lr,)


@app.cell
def _(addiction_labels, addiction_predictions_lr):
    from sklearn.metrics import root_mean_squared_error

    lin_rmse = root_mean_squared_error(addiction_labels, addiction_predictions_lr)
    lin_rmse
    return (root_mean_squared_error,)


@app.cell
def _(addiction, addiction_labels, make_pipeline, preprocessing):
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(addiction, addiction_labels)
    return (tree_reg,)


@app.cell
def _(addiction, addiction_labels, root_mean_squared_error, tree_reg):
    addiction_predictions_dt = tree_reg.predict(addiction)
    tree_rmse = root_mean_squared_error(addiction_labels, addiction_predictions_dt)
    tree_rmse
    return


@app.cell
def _(addiction, addiction_labels, tree_reg):
    from sklearn.model_selection import cross_val_score

    tree_rmses = -cross_val_score(tree_reg, addiction, addiction_labels,
                                  scoring="neg_root_mean_squared_error", cv=10)
    return cross_val_score, tree_rmses


@app.cell
def _(pd, tree_rmses):
    pd.Series(tree_rmses).describe()
    return


@app.cell
def _(
    addiction,
    addiction_labels,
    cross_val_score,
    make_pipeline,
    pd,
    preprocessing,
):
    from sklearn.ensemble import RandomForestRegressor

    random_tree_reg = make_pipeline(preprocessing, RandomForestRegressor(n_estimators=100, random_state=42))
    random_tree_reg.fit(addiction, addiction_labels)

    random_tree_rmses = -cross_val_score(random_tree_reg, addiction, addiction_labels,
                                  scoring="neg_root_mean_squared_error", cv=10)

    pd.Series(random_tree_rmses).describe()
    return


if __name__ == "__main__":
    app.run()

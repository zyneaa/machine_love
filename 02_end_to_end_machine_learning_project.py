# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/zyneaa/colab_ml/blob/main/02_end_to_end_machine_learning_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="Jkht74IV0rtT"
# **Chapter 2 – End-to-end Machine Learning project**

# %% [markdown] id="AlAEYZFw0rtV"
# *This notebook contains all the sample code and solutions to the exercises in chapter 2.*

# %% [markdown] id="SobO2_i-0rtV"
# <table align="left">
#   <td>
#     <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#   </td>
#   <td>
#     <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
#   </td>
# </table>

# %% colab={"base_uri": "https://localhost:8080/"} id="zY9UuV5l0rtV" outputId="318e9435-a48a-473f-e136-dba40d696046"
print("Welcome to Machine Learning!")

# %% [markdown] id="1okpD5h90rtW"
# This project requires Python 3.7 or above:

# %% id="LycFGZY70rtW"
import sys

assert sys.version_info >= (3, 7)

# %% [markdown] id="7yrjL14O0rtW"
# It also requires Scikit-Learn ≥ 1.0.1:

# %% id="G8GmOqYk0rtW"
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
print(sklearn.__version__)

# %% [markdown] id="mvaEmred0rtW"
# # Get the Data

# %% [markdown] id="F4U52RNj0rtX"
# *Welcome to Machine Learning Housing Corp.! Your task is to predict median house values in Californian districts, given a number of features from these districts.*

# %% [markdown] id="oVXzQm650rtX"
# ## Download the Data

# %% colab={"base_uri": "https://localhost:8080/"} id="eIDgSCAd0rtX" outputId="a2c92eb3-3ac5-4d4c-a041-e435e85f24b1"
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# %% [markdown] id="pFaym14d0rtX"
# ## Take a Quick Look at the Data Structure

# %% id="bvj6SE140rtX" outputId="47468911-0e03-4236-9c77-68640f8b1f0c"
housing.head(10)

# %% id="Z4g3x3LT0rtX" outputId="7a778455-62c5-4c55-8c2b-4c28077807aa"
housing.info()

# %% id="CWI_LWzK0rtX" outputId="a1b07160-307d-4dda-d70b-7cdd2245f4a7"
housing["ocean_proximity"].value_counts()

# %% id="54GkZdiE0rtY" outputId="6ddd6569-4efd-41c0-fe9d-97057f47bbbb"
housing.describe()

# %% [markdown] id="j-Tn38Yj0rtY"
# The following cell is not shown either in the book. It creates the `images/end_to_end_project` folder (if it doesn't already exist), and it defines the `save_fig()` function which is used through this notebook to save the figures in high-res for the book.

# %% id="zlMGA1kv0rtY"
import matplotlib.pyplot as plt
# extra code – code to save the figures as high-res PNGs for the book

IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# %% id="csfn9gGO0rtY" outputId="2eff9e0f-8b0a-4971-fa7a-18f93fc23e62"
# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()

# %% [markdown] id="y5y6Yy480rtY"
# ## Create a Test Set

# %% id="2quiynUo0rtY"
import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# %% id="kAxsDmKD0rtY" outputId="bcf3f8ef-e20c-4a40-d20a-990441f1c6bf"
train_set, test_set = shuffle_and_split_data(housing, 0.2)
len(train_set)

# %% id="7IM8B9gk0rtY" outputId="e0e1c548-0bf5-491c-a0f9-fe57e5c7e6d5"
len(test_set)

# %% [markdown] id="EWT4imVl0rtY"
# To ensure that this notebook's outputs remain the same every time we run it, we need to set the random seed:

# %% id="r_jeGN1V0rtY"
np.random.seed(42)

# %% [markdown] id="9pwVRV_E0rtZ"
# Sadly, this won't guarantee that this notebook will output exactly the same results as in the book, since there are other possible sources of variation. The most important is the fact that algorithms get tweaked over time when libraries evolve. So please tolerate some minor differences: hopefully, most of the outputs should be the same, or at least in the right ballpark.

# %% [markdown] id="VlOrqvOU0rtZ"
# Note: another source of randomness is the order of Python sets: it is based on Python's `hash()` function, which is randomly "salted" when Python starts up (this started in Python 3.3, to prevent some denial-of-service attacks). To remove this randomness, the solution is to set the `PYTHONHASHSEED` environment variable to `"0"` _before_ Python even starts up. Nothing will happen if you do it after that. Luckily, if you're running this notebook on Colab, the variable is already set for you.

# %% id="njt6vunV0rtZ"
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# %% id="gcjhAgiM0rtZ"
housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# %% id="LylznbFn0rtZ"
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

# %% id="p4YXercn0rtZ"
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# %% id="ZHC041og0rtZ" outputId="6f84c8c6-75e1-40d5-a26d-58f968b03c20"
test_set["total_bedrooms"].isnull().sum()

# %% [markdown] id="B-aOLcAj0rtZ"
# To find the probability that a random sample of 1,000 people contains less than 48.5% female or more than 53.5% female when the population's female ratio is 51.1%, we use the [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). The `cdf()` method of the binomial distribution gives us the probability that the number of females will be equal or less than the given value.

# %% id="lvSibLS10rta" outputId="0791b677-7761-421d-88df-f83938bd2dd4"
# extra code – shows how to compute the 10.7% proba of getting a bad sample

from scipy.stats import binom

sample_size = 1000
ratio_female = 0.511
proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print(proba_too_small, proba_too_large)
print(proba_too_small + proba_too_large)

# %% [markdown] id="vtuMII6V0rta"
# If you prefer simulations over maths, here's how you could get roughly the same result:

# %% id="ux2dm80P0rta" outputId="eca79966-cd3b-42c6-a1b4-796ffcd6a1c1"
# extra code – shows another way to estimate the probability of bad sample

np.random.seed(42)

samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
((samples < 485) | (samples > 535)).mean()

# %% id="zmM6qhwD0rta"
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# %% id="uaQHXCRd0rtb" outputId="d67d9623-b28c-4286-d745-bcd36fd69af3"
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot")  # extra code
plt.show()

# %% id="Zee9ZgNE0rtb"
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_splits

# %% id="cInNXGzA0rtb"
strat_train_set, strat_test_set = strat_splits[0]

# %% [markdown] id="UdSKMnLT0rtb"
# It's much shorter to get a single stratified split:

# %% id="mpJIeexs0rtb"
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# %% id="0NaL7n060rtb" outputId="f19cc1fc-3179-4561-dcf6-ab7d359b5e76"
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# %% id="nmK1N2rQ0rtb" outputId="f1398b45-c872-4f37-b563-1f1a318fdc71"
from sklearn.model_selection import train_test_split

# extra code – computes the data for Figure 2–10

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)

# %% id="QuoQNeV60rtb"
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# %% [markdown] id="cDuHXHlt0rtc"
# # Discover and Visualize the Data to Gain Insights

# %% id="XbkvomQk0rtc"
housing = strat_train_set.copy()
housing.head()

# %% [markdown] id="9OXWWk_C0rtc"
# ## Visualizing Geographical Data

# %% id="X-0aC8E70rtc" outputId="62bf4da6-e0a8-4938-d5d5-a809d04bd3fd"
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
save_fig("bad_visualization_plot")  # extra code
plt.show()

# %% id="zcWRoMJY0rtc" outputId="675f04f3-7084-40e9-b337-67d6e981a79a"
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
save_fig("better_visualization_plot")  # extra code
plt.show()

# %% id="vPr5Cg0c0rtc" outputId="1a127fcf-7677-43b0-825c-4894cb74ebf1"
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
save_fig("housing_prices_scatterplot")  # extra code
plt.show()

# %% [markdown] id="e71Yt_Bn0rtc"
# The argument `sharex=False` fixes a display bug: without it, the x-axis values and label are not displayed (see: https://github.com/pandas-dev/pandas/issues/10611).

# %% [markdown] id="gUafhLJS0rtc"
# The next cell generates the first figure in the chapter (this code is not in the book). It's just a beautified version of the previous figure, with an image of California added in the background, nicer label names and no grid.

# %% id="1mtkJASY0rtc" outputId="27d8628b-6cbe-45a0-c95f-9e7f61948462"
import ssl

# extra code – this cell generates the first figure in the chapter

# Download the California image
filename = "california.png"
if not (IMAGES_PATH / filename).is_file():
    homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
    url = homl3_root + "images/end_to_end_project/" + filename
    print("Downloading", filename)
    with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as response:
        with open(IMAGES_PATH / filename, 'wb') as f:
            f.write(response.read())

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed.plot(
             kind="scatter", x="Longitude", y="Latitude",
             s=housing_renamed["Population"] / 100, label="Population",
             c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))

california_img = plt.imread(IMAGES_PATH / filename)
axis = -124.55, -113.95, 32.45, 42.05
plt.axis(axis)
plt.imshow(california_img, extent=axis)

save_fig("california_housing_prices_plot")
plt.show()

# %% [markdown] id="iXRNLot30rtc"
# ## Looking for Correlations

# %% [markdown] id="WygujmJP0rtc"
# Note: since Pandas 2.0.0, the `numeric_only` argument defaults to `False`, so we need to set it explicitly to True to avoid an error.

# %% id="gLTIrJvk0rtc"
corr_matrix = housing.corr(numeric_only=True)

# %% id="LP8_dS580rtc" outputId="c42c785d-335d-4613-afb0-d13defbf4d87"
corr_matrix["median_house_value"].sort_values(ascending=False)

# %% id="xgRIWNml0rtc" outputId="3da58984-be90-41a9-e237-0b36112f21e3"
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")  # extra code
plt.show()

# %% id="a4IzKyeT0rtd" outputId="6c7159cb-adc5-4ea2-f59a-1254fff59791"
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
save_fig("income_vs_house_value_scatterplot")  # extra code
print(housing.columns)
plt.show()

# %% [markdown] id="Of6Attva0rtd"
# ## Experimenting with Attribute Combinations

# %% id="xqcQc4cJ0rtd"
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# %% id="vvV9e7xG0rtd" outputId="2558ce67-d277-46f4-cbff-b10e22b97fe9"
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# %% [markdown] id="kPEuUjoJ0rtd"
# # Prepare the Data for Machine Learning Algorithms

# %% [markdown] id="iH41TU0Z0rtd"
# Let's revert to the original training set and separate the target (note that `strat_train_set.drop()` creates a copy of `strat_train_set` without the column, it doesn't actually modify `strat_train_set` itself, unless you pass `inplace=True`):

# %% id="WqZM5xKe0rtd"
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.head()

# %% [markdown] id="zBirGqFD0rtd"
# ## Data Cleaning

# %% [markdown] id="dP3eZGce0rtd"
# In the book 3 options are listed to handle the NaN values:
#
# ```python
# housing.dropna(subset=["total_bedrooms"], inplace=True)    # option 1
#
# housing.drop("total_bedrooms", axis=1)       # option 2
#
# median = housing["total_bedrooms"].median()  # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)
# ```
#
# For each option, we'll create a copy of `housing` and work on that copy to avoid breaking `housing`. We'll also show the output of each option, but filtering on the rows that originally contained a NaN value.

# %% id="44G4j-4y0rtd" outputId="ee3814fd-abca-4da9-fac4-3707f3826feb"
null_rows_idx = housing.isnull().any(axis=1)
housing.loc[null_rows_idx].head()

# %% id="oOVH2dnF0rtd" outputId="a11709aa-dbac-4f61-b275-c76b38187fbb"
housing_option1 = housing.copy()

housing_option1.dropna(subset=["total_bedrooms"], inplace=True)  # option 1

housing_option1.loc[null_rows_idx].head()

# %% id="HATxKKrt0rtd" outputId="55eb9feb-7a94-443e-bab8-54281052a280"
housing_option2 = housing.copy()

housing_option2.drop("total_bedrooms", axis=1, inplace=True)  # option 2

housing_option2.loc[null_rows_idx].head()

# %% id="YG-Vtq9S0rtd" outputId="118970e7-7907-4db1-a686-81bad7c51b3e"
housing_option3 = housing.copy()

median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3

housing_option3.loc[null_rows_idx].head()

# %% id="7wWkJEZ90rtd"
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# %% [markdown] id="YRAm5R5w0rtd"
# Separating out the numerical attributes to use the `"median"` strategy (as it cannot be calculated on text attributes like `ocean_proximity`):

# %% id="jQSUlGmD0rtd"
housing_num = housing.select_dtypes(include=[np.number])

# %% id="8v7fockF0rtd" outputId="d9408675-294d-4671-c003-ead96360e5d8"
imputer.fit(housing_num)

# %% id="U5Zt9VR00rte" outputId="5f4b80c1-b9e6-423e-e124-e6e54682d3d9"
imputer.statistics_

# %% [markdown] id="PhcL2k060rte"
# Check that this is the same as manually computing the median of each attribute:

# %% id="_q-EfSNv0rte" outputId="b1f16c01-c3b9-46b3-8288-7dfcc1e1a0d7"
housing_num.median().values

# %% [markdown] id="qbW4UpUp0rte"
# Transform the training set:

# %% id="krTKOy0R0rte"
X = imputer.transform(housing_num)
X

# %% id="viqXNTW10rte" outputId="1f8efb93-4079-41b4-8aee-fabcc068034d"
imputer.feature_names_in_

# %% id="IkJZDj0b0rte"
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# %% id="q5m2wWFj0rte" outputId="0bd66274-f56f-494b-e9ab-5d980112d685"
housing_tr.loc[null_rows_idx].head()

# %% id="3d93xd2s0rte" outputId="e1e92ada-abfa-46f0-edbe-26f5539b9d55"
imputer.strategy

# %% id="TsMC-NMn0rte"
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# %% id="ByjHg3iH0rte" outputId="6cddece0-b60a-4adc-e9e6-7ba51a0d5713"
housing_tr.loc[null_rows_idx].head()  # not shown in the book

# %% id="UYQVm-Qa0rte"
#from sklearn import set_config
#
# set_config(transform_output="pandas")  # scikit-learn >= 1.2

# %% [markdown] id="mMEeEnPF0rte"
# Now let's drop some outliers:

# %% id="-kKBFfOT0rte"
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)

# %% id="29c33xH-0rte" outputId="a09b3909-fd52-49b4-b61d-95f8542e630d"
outlier_pred

# %% [markdown] id="z2ErJy2g0rte"
# If you wanted to drop outliers, you would run the following code:

# %% id="0YqumCsY0rte"
#housing = housing.iloc[outlier_pred == 1]
#housing_labels = housing_labels.iloc[outlier_pred == 1]

# %% [markdown] id="vIIx13NU0rte"
# ## Handling Text and Categorical Attributes

# %% [markdown] id="gS82wCTk0rte"
# Now let's preprocess the categorical input feature, `ocean_proximity`:

# %% id="szs3Jfap0rtf" outputId="267e35ac-8d0f-4c7a-d8a3-8614c6949fef"
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(8)

# %% id="UQzUkipP0rtf"
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# %% id="Ruxn_Q9a0rtf" outputId="236f4bc4-557e-4db8-d087-515598266d91"
housing_cat_encoded[:8]

# %% id="gqOT_q_m0rtf" outputId="cdeabbde-6be7-436c-e1ac-ef86318e290d"
ordinal_encoder.categories_

# %% id="lXeDSNtL0rtf"
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# %% id="fjC0wRW30rtf" outputId="5759728e-534d-41e2-f7c2-67250c8cfee1"
housing_cat_1hot

# %% [markdown] id="x4f6a4uk0rtf"
# By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if needed by calling the `toarray()` method:

# %% id="5n1_e21K0rtf" outputId="bc89f881-29d6-4907-c109-c357ddb18107"
housing_cat_1hot.toarray()

# %% [markdown] id="kl_DTsRq0rtf"
# Alternatively, you can set `sparse_output=False` when creating the `OneHotEncoder` (note: the `sparse` hyperparameter was renamned to `sparse_output` in Scikit-Learn 1.2):

# %% id="2dWKFpc00rtf" outputId="5ff2d58f-3f38-4967-84c1-69113a04ea0d"
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# %% id="yias56dQ0rtg" outputId="5c0d552c-9e8c-41c9-df32-4c3c3daefe0a"
cat_encoder.categories_

# %% id="_dN300gM0rtg" outputId="ba6f5c6b-0c62-424f-9096-ce3eac2ef86d"
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)

# %% id="Sxv_XfSz0rtg" outputId="9e808768-6834-4b19-ccbc-e2abf4400d9c"
cat_encoder.transform(df_test)

# %% id="3RD0SbLp0rtg" outputId="2f6ee0b1-7f06-4871-8f62-4330d9011473"
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)

# %% id="M-izgq190rtg" outputId="8e1ef1d9-0bcf-4bc5-8b84-49f54cb70c07"
cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)

# %% id="-Huegli40rtg" outputId="caab25c5-6fad-447a-834e-cf6b683d2a4f"
cat_encoder.feature_names_in_

# %% id="Wqsy9cCL0rtg" outputId="0a562892-fccf-480f-85d3-0f8d17bc773e"
cat_encoder.get_feature_names_out()

# %% id="bmC_JcKw0rtg"
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)

# %% id="O3hpqAnR0rtg" outputId="734d44ee-3c78-4d4b-d873-4a54c1ffab81"
df_output

# %% [markdown] id="dzJU2NjC0rtg"
# ## Feature Scaling

# %% id="YTlFoAzR0rtg"
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# %% id="OBWvXiUb0rtg"
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# %% id="JjLqzYDq0rtg" outputId="3434b4fe-f5ef-4ab0-e109-68ecb01f359b"
# extra code – this cell generates Figure 2–17
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Population")
axs[1].set_xlabel("Log of population")
axs[0].set_ylabel("Number of districts")
save_fig("long_tail_plot")
plt.show()

# %% [markdown] id="2ZGKSDVs0rtg"
# What if we replace each value with its percentile?

# %% id="v9OHeuXv0rtg" outputId="655846aa-606d-4855-c7c5-6f49fc8ec433"
# extra code – just shows that we get a uniform distribution
percentiles = [np.percentile(housing["median_income"], p)
               for p in range(1, 100)]
flattened_median_income = pd.cut(housing["median_income"],
                                 bins=[-np.inf] + percentiles + [np.inf],
                                 labels=range(1, 100 + 1))
flattened_median_income.hist(bins=50)
plt.xlabel("Median income percentile")
plt.ylabel("Number of districts")
plt.show()
# Note: incomes below the 1st percentile are labeled 1, and incomes above the
# 99th percentile are labeled 100. This is why the distribution below ranges
# from 1 to 100 (not 0 to 100).

# %% id="B72Ypv_Y0rtg"
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# %% id="SGYmFRTo0rtg" outputId="d3c39c77-5ded-4d10-f995-d30303fb7934"
# extra code – this cell generates Figure 2–18

ages = np.linspace(housing["housing_median_age"].min(),
                   housing["housing_median_age"].max(),
                   500).reshape(-1, 1)
gamma1 = 0.1
gamma2 = 0.03
rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Housing median age")
ax1.set_ylabel("Number of districts")
ax1.hist(housing["housing_median_age"], bins=50)

ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
color = "blue"
ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel("Age similarity", color=color)

plt.legend(loc="upper left")
save_fig("age_similarity_plot")
plt.show()

# %% id="l6E4F8rm0rth"
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# %% id="1KbZB6Kj0rth" outputId="de13209c-2ced-4fef-98c8-4e2e4213d2f8"
predictions

# %% id="_hBe18et0rth"
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# %% id="-p_f0GuE0rth" outputId="711a74bb-f57c-4970-c3c0-28d8604f56fa"
predictions

# %% [markdown] id="VYl1wdIH0rth"
# ## Custom Transformers

# %% [markdown] id="GJrJPxL90rth"
# To create simple transformers:

# %% id="O6fczjcL0rth"
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# %% id="ugYMlxZD0rth"
rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# %% id="_15NSuGS0rth" outputId="1ea5d3ce-5015-4b77-bff5-8f457769feec"
age_simil_35

# %% id="0VofAPQq0rth"
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

# %% id="qh8gqBYK0rth" outputId="3cda43d5-3d21-4a3a-ccd2-c07eaf6c65ce"
sf_simil

# %% id="03tZB3pM0rth" outputId="18ecfc52-b3af-4c07-f315-b972de955e77"
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))

# %% id="yXPY7RU60rth"
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


# %% id="bCoHZPVf0rth"
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# %% [markdown] id="AuLL_fQO0rth"
# **Warning**:
# * There was a change in Scikit-Learn 1.3.0 which affected the random number generator for `KMeans` initialization. Therefore the results will be different than in the book if you use Scikit-Learn ≥ 1.3. That's not a problem as long as you don't expect the outputs to be perfectly identical.
# * Throughout this notebook, when `n_init` was not set when creating a `KMeans` estimator, I explicitly set it to `n_init=10` to avoid a warning about the fact that the default value for this hyperparameter will change from 10 to `"auto"` in Scikit-Learn 1.4.
# * The book was unclear about the fact that setting `sample_weight=housing_labels` was only meant as an example, it's not actually used during training. So I remove the `sample_weight` argument below, and the next figure corresponds to the clusters actually used during training (unlike Figure 2-19 in the book). Sorry if this caused any confusion!

# %% id="AesUOADF0rth"
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])

# %% id="gal1zFF20rth" outputId="4b78c477-7e00-4d0e-c6f9-4c31807d92ff"
similarities[:3].round(2)

# %% id="BdMso2eV0rth" outputId="880dbea3-59fd-4057-c411-b8325c20f070"
# extra code – this cell generates Figure 2–19

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"] / 100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")
save_fig("district_cluster_plot")
plt.show()

# %% [markdown] id="vAQZorV_0rti"
# ## Transformation Pipelines

# %% [markdown] id="Y4N4LYGD0rti"
# Now let's build a pipeline to preprocess the numerical attributes:

# %% id="BnrR1Au80rti"
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

# %% id="XYI8uVOU0rti"
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# %% id="Nnvw0ac50rti" outputId="1afb7b99-3598-4eb4-fcad-4da83abe3c11"
from sklearn import set_config

set_config(display='diagram')

num_pipeline

# %% id="A_BChhA90rti" outputId="76d3760b-c470-4062-997c-dfb046acde73"
housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)


# %% id="iVVj-1wm0rti"
def monkey_patch_get_signature_names_out():
    """Monkey patch some classes which did not handle get_feature_names_out()
       correctly in Scikit-Learn 1.0.*."""
    from inspect import Signature, signature, Parameter
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
      print("Monkey-patching SimpleImputer.get_feature_names_out()")
      SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out

monkey_patch_get_signature_names_out()

# %% id="ly-qnBkl0rti"
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)

# %% id="I7yKJDsM0rti" outputId="9a213c1f-4227-4a1a-82ec-d461ac8955c6"
df_housing_num_prepared.head(2)  # extra code

# %% id="unjTg3GB0rti" outputId="e0df9adc-588a-469f-cf6e-ce5556a80401"
num_pipeline.steps

# %% id="dWbB0T1a0rti" outputId="9aed9b16-f0b4-4a16-e99d-5d8369993674"
num_pipeline[1]

# %% id="8Dzv_tYH0rti" outputId="cfeb0acc-db96-4fb9-82fd-c4e97ea159fe"
num_pipeline[:-1]

# %% id="HbR00A4d0rti" outputId="497e5bba-e983-4497-d83f-a5c7dde78412"
num_pipeline.named_steps["simpleimputer"]

# %% id="Nlcy_S200rti" outputId="271cb699-b6cf-467b-acfa-f62f94f0de63"
num_pipeline.set_params(simpleimputer__strategy="median")

# %% id="05uOCE_00rti"
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# %% id="FIMFAatS0rti"
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

# %% id="xv4dXX0G0rti"
housing_prepared = preprocessing.fit_transform(housing)

# %% id="zFe6xO5W0rti" outputId="5fa2a8e9-7181-4a3c-ce48-cc8dc2c14ac6"
# extra code – shows that we can get a DataFrame out if we want
housing_prepared_fr = pd.DataFrame(
    housing_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=housing.index)
housing_prepared_fr.head(2)


# %% id="KPq8yRpV0rti"
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

# %% id="HaZImZiJ0rti" outputId="22dea5e5-21b4-48db-f597-c963551fd78a"
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape

# %% id="Mh5BVSX-0rtj" outputId="e93d86c5-900c-400e-8987-08ded78d6b20"
preprocessing.get_feature_names_out()

# %% [markdown] id="zNlDoldX0rtj"
# # Select and Train a Model

# %% [markdown] id="GJpa26wt0rtj"
# ## Training and Evaluating on the Training Set

# %% id="YVIRdhNf0rtj" outputId="0256a4ae-ffba-4986-8fac-47b81c100b53"
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# %% [markdown] id="4ZvnWYHq0rtj"
# Let's try the full preprocessing pipeline on a few training instances:

# %% id="MBo6RQO50rtj" outputId="8c88a8f2-3e6a-4825-e59e-1580b0377abe"
housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2)  # -2 = rounded to the nearest hundred

# %% [markdown] id="f8MpXgvj0rtj"
# Compare against the actual values:

# %% id="zEXyB6tB0rtj" outputId="1a14af1f-995e-4719-a0f2-0d9ea1374cc7"
housing_labels.iloc[:5].values

# %% id="IyFQorsa0rtj" outputId="216b63f9-8cba-49a7-b78a-44cb663f8a1a"
# extra code – computes the error ratios discussed in the book
error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

# %% [markdown] id="iycj_sJi0rtj"
# **Warning**: In recent versions of Scikit-Learn, you must use `root_mean_squared_error(labels, predictions)` to compute the RMSE, instead of `mean_squared_error(labels, predictions, squared=False)`. The following `try`/`except` block tries to import `root_mean_squared_error`, and if it fails it just defines it.

# %% id="eSdFpJyQ0rtj" outputId="bd8452d9-67e9-4a0c-e9b2-929adcaef249"
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
lin_rmse

# %% id="3XxfHiKl0rtj" outputId="a021ec65-0b52-4afb-ad99-43bec934dc3d"
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

# %% id="EoEKr16E0rtj" outputId="547ffdb1-c899-45a7-ff13-ea3ab49671d6"
housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
tree_rmse

# %% [markdown] id="6wxrfNZA0rtj"
# ## Better Evaluation Using Cross-Validation

# %% id="SWoMnOAK0rtj"
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)

# %% id="Dqc-f7JY0rtj" outputId="431da9a5-4fec-4d03-ad88-a6e70d375f30"
pd.Series(tree_rmses).describe()

# %% id="-A0yIqzd0rtj" outputId="015fb02d-35cb-4453-f0e3-b817439a5a40"
# extra code – computes the error stats for the linear model
lin_rmses = -cross_val_score(lin_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_rmses).describe()

# %% [markdown] id="uSt8hD5k0rtj"
# **Warning:** the following cell may take a few minutes to run:

# %% id="DwLUJY2q0rtj"
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)

# %% id="yYfh3qs-0rtj" outputId="e6e437ed-dfe5-46f2-fb91-312d68a6c7f9"
pd.Series(forest_rmses).describe()

# %% [markdown] id="EfvyDYVQ0rtj"
# Let's compare this RMSE measured using cross-validation (the "validation error") with the RMSE measured on the training set (the "training error"):

# %% id="d0oKZrfp0rtj" outputId="10ae14ee-84eb-40f5-90bd-9a89d136d292"
forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)
forest_rmse = root_mean_squared_error(housing_labels, housing_predictions)
forest_rmse

# %% [markdown] id="s-jwhAOV0rtk"
# The training error is much lower than the validation error, which usually means that the model has overfit the training set. Another possible explanation may be that there's a mismatch between the training data and the validation data, but it's not the case here, since both came from the same dataset that we shuffled and split in two parts.

# %% [markdown] id="RHWkPg_c0rtk"
# # Fine-Tune Your Model

# %% [markdown] id="Si96Q8K-0rtk"
# ## Grid Search

# %% [markdown] id="6VKDrj3j0rtk"
# **Warning:** the following cell may take a few minutes to run:

# %% id="OlKn2Liz0rtk" outputId="d21a2a80-ade8-4374-e22f-3b5a7bb1ad2f"
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)

# %% [markdown] id="3F7K-RyU0rtk"
# You can get the full list of hyperparameters available for tuning by looking at `full_pipeline.get_params().keys()`:

# %% id="j0qp3TfQ0rtk" outputId="55c6440c-d424-4994-a030-b8da13123639"
# extra code – shows part of the output of get_params().keys()
print(str(full_pipeline.get_params().keys())[:1000] + "...")

# %% [markdown] id="HA0cOsao0rtk"
# The best hyperparameter combination found:

# %% id="7HOfPyDE0rtk" outputId="fc8fa710-8c88-4d0c-e62e-74182ec46f76"
grid_search.best_params_

# %% id="axjVAyes0rtk" outputId="de68702b-4dbe-4222-8e84-6b5da04500a3"
grid_search.best_estimator_

# %% [markdown] id="GkbWVJkF0rtk"
# Let's look at the score of each hyperparameter combination tested during the grid search:

# %% id="dVcseJou0rtk" outputId="cd2aa074-aa82-4d02-daf6-d78457c96099"
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# extra code – these few lines of code just make the DataFrame look nicer
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

cv_res.head()

# %% [markdown] id="NJSA0gTv0rtk"
# ## Randomized Search

# %% id="_HoIKOrm0rtk"
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# %% [markdown] id="eu1vwvTC0rtk"
# Try 30 (`n_iter` × `cv`) random combinations of hyperparameters:

# %% [markdown] id="FL6zIwKG0rtk"
# **Warning:** the following cell may take a few minutes to run:

# %% id="27CNvMqI0rtk" outputId="22398466-576f-4612-ac51-05e23d77f6a1"
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)

# %% id="TNwdb4tP0rtk" outputId="b4bebf2b-43ea-477c-8e5c-08b99752def0"
# extra code – displays the random search results
cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()

# %% [markdown] id="Fs0un3qz0rtk"
# **Bonus section: how to choose the sampling distribution for a hyperparameter**
#
# * `scipy.stats.randint(a, b+1)`: for hyperparameters with _discrete_ values that range from a to b, and all values in that range seem equally likely.
# * `scipy.stats.uniform(a, b)`: this is very similar, but for _continuous_ hyperparameters.
# * `scipy.stats.geom(1 / scale)`: for discrete values, when you want to sample roughly in a given scale. E.g., with scale=1000 most samples will be in this ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
# * `scipy.stats.expon(scale)`: this is the continuous equivalent of `geom`. Just set `scale` to the most likely value.
# * `scipy.stats.loguniform(a, b)`: when you have almost no idea what the optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then you're just as likely to sample a value between 0.01 and 0.1 as a value between 10 and 100.
#

# %% [markdown] id="5lSzlOyb0rtk"
# Here are plots of the probability mass functions (for discrete variables), and probability density functions (for continuous variables) for `randint()`, `uniform()`, `geom()` and `expon()`:

# %% id="OWaBX15c0rtk" outputId="046f1b70-7ed0-4182-bdc2-65efe77125fd"
# extra code – plots a few distributions you can use in randomized search

from scipy.stats import randint, uniform, geom, expon

xs1 = np.arange(0, 7 + 1)
randint_distrib = randint(0, 7 + 1).pmf(xs1)

xs2 = np.linspace(0, 7, 500)
uniform_distrib = uniform(0, 7).pdf(xs2)

xs3 = np.arange(0, 7 + 1)
geom_distrib = geom(0.5).pmf(xs3)

xs4 = np.linspace(0, 7, 500)
expon_distrib = expon(scale=1).pdf(xs4)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.bar(xs1, randint_distrib, label="scipy.randint(0, 7 + 1)")
plt.ylabel("Probability")
plt.legend()
plt.axis([-1, 8, 0, 0.2])

plt.subplot(2, 2, 2)
plt.fill_between(xs2, uniform_distrib, label="scipy.uniform(0, 7)")
plt.ylabel("PDF")
plt.legend()
plt.axis([-1, 8, 0, 0.2])

plt.subplot(2, 2, 3)
plt.bar(xs3, geom_distrib, label="scipy.geom(0.5)")
plt.xlabel("Hyperparameter value")
plt.ylabel("Probability")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.subplot(2, 2, 4)
plt.fill_between(xs4, expon_distrib, label="scipy.expon(scale=1)")
plt.xlabel("Hyperparameter value")
plt.ylabel("PDF")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.show()

# %% [markdown] id="GcfgoLKM0rtl"
# Here are the PDF for `expon()` and `loguniform()` (left column), as well as the PDF of log(X) (right column). The right column shows the distribution of hyperparameter _scales_. You can see that `expon()` favors hyperparameters with roughly the desired scale, with a longer tail towards the smaller scales. But `loguniform()` does not favor any scale, they are all equally likely:

# %% id="Ce3dHtUu0rtl" outputId="b0917f7f-06df-400f-f076-b17d85f43900"
# extra code – shows the difference between expon and loguniform

from scipy.stats import loguniform

xs1 = np.linspace(0, 7, 500)
expon_distrib = expon(scale=1).pdf(xs1)

log_xs2 = np.linspace(-5, 3, 500)
log_expon_distrib = np.exp(log_xs2 - np.exp(log_xs2))

xs3 = np.linspace(0.001, 1000, 500)
loguniform_distrib = loguniform(0.001, 1000).pdf(xs3)

log_xs4 = np.linspace(np.log(0.001), np.log(1000), 500)
log_loguniform_distrib = uniform(np.log(0.001), np.log(1000)).pdf(log_xs4)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.fill_between(xs1, expon_distrib,
                 label="scipy.expon(scale=1)")
plt.ylabel("PDF")
plt.legend()
plt.axis([0, 7, 0, 1])

plt.subplot(2, 2, 2)
plt.fill_between(log_xs2, log_expon_distrib,
                 label="log(X) with X ~ expon")
plt.legend()
plt.axis([-5, 3, 0, 1])

plt.subplot(2, 2, 3)
plt.fill_between(xs3, loguniform_distrib,
                 label="scipy.loguniform(0.001, 1000)")
plt.xlabel("Hyperparameter value")
plt.ylabel("PDF")
plt.legend()
plt.axis([0.001, 1000, 0, 0.005])

plt.subplot(2, 2, 4)
plt.fill_between(log_xs4, log_loguniform_distrib,
                 label="log(X) with X ~ loguniform")
plt.xlabel("Log of hyperparameter value")
plt.legend()
plt.axis([-8, 1, 0, 0.2])

plt.show()

# %% [markdown] id="Ybq2p6k20rtl"
# ## Analyze the Best Models and Their Errors

# %% id="ZEhMpHcd0rtl" outputId="067978c9-18f2-4b31-ccba-fb0e07eb54b0"
final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)

# %% id="f8DBpXZQ0rtl" outputId="ab301271-1ae6-4117-9407-a74360e6ce2f"
sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
           reverse=True)

# %% [markdown] id="JX_T8K7s0rtl"
# ## Evaluate Your System on the Test Set

# %% id="Dxd7fNMF0rtl" outputId="671ee73f-fb2a-4fcd-be49-26e31f877e02"
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)

# %% [markdown] id="ltbTJlbZ0rtl"
# We can compute a 95% confidence interval for the test RMSE:

# %% id="SlpCUVys0rtl"
from scipy import stats

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
                              confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval


# %% id="VcgEhQ580rtl" outputId="0552aabf-a447-4a34-ef82-2bda53eba389"
rmse_lower, rmse_upper

# %% [markdown] id="67bzhTjE0rtl"
# ## Model persistence using joblib

# %% [markdown] id="hV8BS1od0rtl"
# Save the final model:

# %% id="BfpvdooL0rtl" outputId="ff1dd97e-2bdf-493a-ec03-7204069444fa"
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")

# %% [markdown] id="gSWj1Nx50rtl"
# Now you can deploy this model to production. For example, the following code could be a script that would run in production:

# %% id="nnHEKcp00rtl"
import joblib

# extra code – excluded for conciseness
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

#class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]

final_model_reloaded = joblib.load("my_california_housing_model.pkl")

new_data = housing.iloc[:5]  # pretend these are new districts
predictions = final_model_reloaded.predict(new_data)

# %% id="YIa9EZ6M0rtl" outputId="25a79cf5-fbf9-46a4-ed48-5158520fb709"
predictions

# %% [markdown] id="1tZAu-u60rtl"
# You could use pickle instead, but joblib is more efficient.

# %% [markdown] id="tuZbWGSu0rtl"
# # Exercise solutions

# %% [markdown] id="bVZKJ4c80rtl"
# ## 1.

# %% [markdown] id="RxPAd4WB0rtl"
# Exercise: _Try a Support Vector Machine regressor (`sklearn.svm.SVR`) with various hyperparameters, such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Note that SVMs don't scale well to large datasets, so you should probably train your model on just the first 5,000 instances of the training set and use only 3-fold cross-validation, or else it will take hours. Don't worry about what the hyperparameters mean for now (see the SVM notebook if you're interested). How does the best `SVR` predictor perform?_

# %% id="tqDg3Aan0rtl" outputId="cc18f083-e441-45c2-be7a-e85965dae1d4"
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
        {'svr__kernel': ['linear'], 'svr__C': [10., 30., 100., 300., 1000.,
                                               3000., 10000., 30000.0]},
        {'svr__kernel': ['rbf'], 'svr__C': [1.0, 3.0, 10., 30., 100., 300.,
                                            1000.0],
         'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# %% [markdown] id="bzq0wVhM0rtl"
# The best model achieves the following score (evaluated using 3-fold cross validation):

# %% id="1zr4_qFK0rtm" outputId="4703ffd5-8c4a-4122-b0ce-5c3c4ad5b668"
svr_grid_search_rmse = -grid_search.best_score_
svr_grid_search_rmse

# %% [markdown] id="vmjJEwPF0rtm"
# That's much worse than the `RandomForestRegressor` (but to be fair, we trained the model on much less data). Let's check the best hyperparameters found:

# %% id="EOuPkR-G0rtm" outputId="5262d4c0-af63-449e-d3cb-25d9ebd7ad2c"
grid_search.best_params_

# %% [markdown] id="VLEEQtzd0rtm"
# The linear kernel seems better than the RBF kernel. Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

# %% [markdown] id="WZSUl9Zl0rtm"
# ## 2.

# %% [markdown] id="UxdLuRf10rtm"
# Exercise: _Try replacing the `GridSearchCV` with a `RandomizedSearchCV`._

# %% [markdown] id="h258_hPF0rtm"
# **Warning:** the following cell will take several minutes to run. You can specify `verbose=2` when creating the `RandomizedSearchCV` if you want to see the training details.

# %% id="KIGQnTXU0rtm" outputId="5da65669-e9b6-4da8-daea-c62f3074a4bf"
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, loguniform

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `loguniform()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': loguniform(20, 200_000),
        'svr__gamma': expon(scale=1.0),
    }

rnd_search = RandomizedSearchCV(svr_pipeline,
                                param_distributions=param_distribs,
                                n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=42)
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# %% [markdown] id="wABUOo2i0rtm"
# The best model achieves the following score (evaluated using 3-fold cross validation):

# %% id="w7cZ7H-A0rtm" outputId="2c830167-4dda-4b52-e882-9ab074a4f1ed"
svr_rnd_search_rmse = -rnd_search.best_score_
svr_rnd_search_rmse

# %% [markdown] id="0oiG7pLG0rtn"
# Now that's really much better, but still far from the `RandomForestRegressor`'s performance. Let's check the best hyperparameters found:

# %% id="WXCSchNw0rtn" outputId="3be52455-ec93-451c-a9ed-c8cb123dfa65"
rnd_search.best_params_

# %% [markdown] id="hP2yO_Ck0rtn"
# This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.

# %% [markdown] id="p3WeR8KO0rtn"
# Note that we used the `expon()` distribution for `gamma`, with a scale of 1, so `RandomSearch` mostly searched for values roughly of that scale: about 80% of the samples were between 0.1 and 2.3 (roughly 10% were smaller and 10% were larger):

# %% id="eUmWvCZE0rtn" outputId="952fb72a-b2fb-4c99-fe04-c377fe484875"
np.random.seed(42)

s = expon(scale=1).rvs(100_000)  # get 100,000 samples
((s > 0.105) & (s < 2.29)).sum() / 100_000

# %% [markdown] id="OwxQQ_Ld0rtn"
# We used the `loguniform()` distribution for `C`, meaning we did not have a clue what the optimal scale of `C` was before running the random search. It explored the range from 20 to 200 just as much as the range from 2,000 to 20,000 or from 20,000 to 200,000.

# %% [markdown] id="DVxjjBB70rtn"
# ## 3.

# %% [markdown] id="pqv8WRQK0rtn"
# Exercise: _Try adding a `SelectFromModel` transformer in the preparation pipeline to select only the most important attributes._

# %% [markdown] id="FTftKByZ0rtn"
# Let's create a new pipeline that runs the previously defined preparation pipeline, and adds a `SelectFromModel` transformer based on a `RandomForestRegressor` before the final regressor:

# %% id="7l_-0x1x0rtn"
from sklearn.feature_selection import SelectFromModel

selector_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),  # min feature importance
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])

# %% id="TaPdHVFP0rtn" outputId="e566d2cf-adfe-4725-c375-6f8e22186274"
selector_rmses = -cross_val_score(selector_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
pd.Series(selector_rmses).describe()

# %% [markdown] id="aBlnQDNh0rtn"
# Oh well, feature selection does not seem to help. But maybe that's just because the threshold we used was not optimal. Perhaps try tuning it using random search or grid search?

# %% [markdown] id="J4fp1ggM0rtn"
# ## 4.

# %% [markdown] id="Jef4U9ub0rtn"
# Exercise: _Try creating a custom transformer that trains a k-Nearest Neighbors regressor (`sklearn.neighbors.KNeighborsRegressor`) in its `fit()` method, and outputs the model's predictions in its `transform()` method. Then add this feature to the preprocessing pipeline, using latitude and longitude as the inputs to this transformer. This will add a feature in the model that corresponds to the housing median price of the nearest districts._

# %% [markdown] id="cRzygT1_0rtn"
# Rather than restrict ourselves to k-Nearest Neighbors regressors, let's create a transformer that accepts any regressor. For this, we can extend the `MetaEstimatorMixin` and have a required `estimator` argument in the constructor. The `fit()` method must work on a clone of this estimator, and it must also save `feature_names_in_`. The `MetaEstimatorMixin` will ensure that `estimator` is listed as a required parameters, and it will update `get_params()` and `set_params()` to make the estimator's hyperparameters available for tuning. Lastly, we create a `get_feature_names_out()` method: the output column name is the ...

# %% id="3xXcFRGc0rtn"
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.utils.validation import check_array

class FeatureFromRegressor(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        check_array(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)
        estimator_class_name = self.estimator_.__class__.__name__
        estimator_short_name = estimator_class_name.lower().replace("_", "")
        return [f"{estimator_short_name}_prediction_{i}"
                for i in range(n_outputs)]


# %% [markdown] id="kw-rlHOZ0rtn"
# Let's ensure it complies to Scikit-Learn's API:

# %% id="UaMBK_8Q0rtn"
from sklearn.utils.estimator_checks import check_estimator

test_results = check_estimator(FeatureFromRegressor(KNeighborsRegressor()))

# %% [markdown] id="t3RSE1FE0rtn"
# Good! Now let's test it:

# %% id="t0HHdkvD0rtn" outputId="5b7eb4bc-a6f0-49bf-befa-0f6118a220e1"
knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = FeatureFromRegressor(knn_reg)
geo_features = housing[["latitude", "longitude"]]
knn_transformer.fit_transform(geo_features, housing_labels)

# %% [markdown] id="bzATuYml0rtn"
# And what does its output feature name look like?

# %% id="HYvCwyBi0rtn" outputId="a70ed444-f20c-4901-a828-827110e69b03"
knn_transformer.get_feature_names_out()

# %% [markdown] id="6imV3-ks0rtn"
# Okay, now let's include this transformer in our preprocessing pipeline:

# %% id="Avc6AuFQ0rto"
from sklearn.base import clone

transformers = [(name, clone(transformer), columns)
                for name, transformer, columns in preprocessing.transformers]
geo_index = [name for name, _, _ in transformers].index("geo")
transformers[geo_index] = ("geo", knn_transformer, ["latitude", "longitude"])

new_geo_preprocessing = ColumnTransformer(transformers)

# %% id="6Yq64DTm0rto"
new_geo_pipeline = Pipeline([
    ('preprocessing', new_geo_preprocessing),
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])

# %% id="KJ_MfKzu0rto" outputId="71da9079-7dd4-4bc9-ece5-de17c881ae5c"
new_pipe_rmses = -cross_val_score(new_geo_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
pd.Series(new_pipe_rmses).describe()

# %% [markdown] id="M8iKOvq90rto"
# Yikes, that's terrible! Apparently the cluster similarity features were much better. But perhaps we should tune the `KNeighborsRegressor`'s hyperparameters? That's what the next exercise is about.

# %% [markdown] id="dg7KviDC0rto"
# ## 5.

# %% [markdown] id="4fUhI8280rto"
# Exercise: _Automatically explore some preparation options using `RandomSearchCV`._

# %% id="QIgxIIm20rto" outputId="26d9e634-8b8b-465a-ba63-ca3aa0768fb2"
param_distribs = {
    "preprocessing__geo__estimator__n_neighbors": range(1, 30),
    "preprocessing__geo__estimator__weights": ["distance", "uniform"],
    "svr__C": loguniform(20, 200_000),
    "svr__gamma": expon(scale=1.0),
}

new_geo_rnd_search = RandomizedSearchCV(new_geo_pipeline,
                                        param_distributions=param_distribs,
                                        n_iter=50,
                                        cv=3,
                                        scoring='neg_root_mean_squared_error',
                                        random_state=42)
new_geo_rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# %% id="gFMfqUoY0rto" outputId="62538bd6-3bc6-4e37-914c-cfad22a92a26"
new_geo_rnd_search_rmse = -new_geo_rnd_search.best_score_
new_geo_rnd_search_rmse

# %% [markdown] id="64uHT8zc0rto"
# Oh well... at least we tried! It looks like the cluster similarity features are definitely better than the KNN feature. But perhaps you could try having both? And maybe training on the full training set would help as well.

# %% [markdown] id="7OS7Eett0rto"
# ## 6.

# %% [markdown] id="yVtR1sT00rto"
# Exercise: _Try to implement the `StandardScalerClone` class again from scratch, then add support for the `inverse_transform()` method: executing `scaler.inverse_transform(scaler.fit_transform(X))` should return an array very close to `X`. Then add support for feature names: set <code>feature_names_in&#95;</code> in the `fit()` method if the input is a DataFrame. This attribute should be a NumPy array of column names. Lastly, implement the `get_feature_names_out()` method: it should have one optional `input_features=None` argument. If passed, the method should check that its length matches <code>n_features_in&#95;</code>, and it should match <code>feature_names_in&#95;</code> if it is defined, then `input_features` should be returned. If `input_features` is `None`, then the method should return <code>feature_names_in&#95;</code> if it is defined or `np.array(["x0", "x1", ...])` with length <code>n_features_in&#95;</code> otherwise._

# %% id="5voPU4Lq0rto"
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class StandardScalerClone(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1  # Avoid division by zero
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return getattr(self, "feature_names_in_",
                           [f"x{i}" for i in range(self.n_features_in_)])
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Invalid number of features")
            if hasattr(self, "feature_names_in_") and not np.all(
                self.feature_names_in_ == input_features
            ):
                raise ValueError("input_features ≠ feature_names_in_")
            return input_features


# %% [markdown] id="SbloULwr0rto"
# Let's test our custom transformer:

# %% id="LPmAPr9N0rto"
from sklearn.utils.estimator_checks import check_estimator

check_estimator(StandardScalerClone())

# %% [markdown] id="rj9zdc4H0rto"
# No errors, that's a great start, we respect the Scikit-Learn API.

# %% [markdown] id="_Dfw75T70rto"
# Now let's ensure the transformation works as expected:

# %% id="2v3sXz_j0rto"
np.random.seed(42)
X = np.random.rand(1000, 3)

scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(X)

assert np.allclose(X_scaled, (X - X.mean(axis=0)) / X.std(axis=0))

# %% [markdown] id="45BHz3Ba0rto"
# How about setting `with_mean=False`?

# %% id="7_0gGKFY0rto"
scaler = StandardScalerClone(with_mean=False)
X_scaled_uncentered = scaler.fit_transform(X)

assert np.allclose(X_scaled_uncentered, X / X.std(axis=0))

# %% [markdown] id="lXGor4lL0rto"
# And does the inverse work?

# %% id="CkihjfOa0rto"
scaler = StandardScalerClone()
X_back = scaler.inverse_transform(scaler.fit_transform(X))

assert np.allclose(X, X_back)

# %% [markdown] id="pmAY9kpV0rto"
# How about the feature names out?

# %% id="SjxRHCRa0rto"
assert np.all(scaler.get_feature_names_out() == ["x0", "x1", "x2"])
assert np.all(scaler.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c"])

# %% [markdown] id="wreMacsn0rtp"
# And if we fit a DataFrame, are the feature in and out ok?

# %% id="gNyaYdUy0rtp"
df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(df)

assert np.all(scaler.feature_names_in_ == ["a", "b"])
assert np.all(scaler.get_feature_names_out() == ["a", "b"])

# %% [markdown] id="_EaGZVbt0rtp"
# All good! That's all for today! 😀

# %% [markdown] id="kLPA4pB60rtp"
# Congratulations! You already know quite a lot about Machine Learning. :)

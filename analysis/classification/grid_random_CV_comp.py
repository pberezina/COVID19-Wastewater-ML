from pathlib import Path

import numpy as np
import click
from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

try:
    from utils import DATA_PATH
except ImportError:  # notebook being ran in child dir
    import sys

    sys.path.insert(0, "..")  # add parent to path
    from utils import DATA_PATH

from models import CLASSIFIERS


# grab a classifier
clf = CLASSIFIERS["knn"]["model"]
param_grid = CLASSIFIERS["knn"]["param_grid"]
rng = np.random.RandomState(0)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}\n".format(results["params"][candidate]))


@click.command()
@click.option(
    "-f",
    "--data_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(DATA_PATH),
    help="Path to data file csv.",
)
def main(data_file: Path) -> None:
    df = load_data(data_file)
    x_cols = ["Confirmed Inmate Deaths", "Recovered Inmates"]
    y_col = "housing_factor"
    X = df[x_cols]
    y = df[y_col]

    search_hyperparameters(X, y)


def load_data(data_file: Path) -> pd.DataFrame:
    df = pd.read_csv(data_file, parse_dates=["Date"])
    return df


def search_hyperparameters(X, y):
    # run randomized search
    n_iter_search = 15
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_grid, n_iter=n_iter_search
    )

    start = time()
    random_search.fit(X, y)
    print(
        "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
        % ((time() - start), n_iter_search)
    )
    report(random_search.cv_results_)

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X, y)

    print(
        "GridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (time() - start, len(grid_search.cv_results_["params"]))
    )
    report(grid_search.cv_results_)

    # run successive halving
    start = time()
    gsh = HalvingGridSearchCV(
        estimator=clf, param_grid=param_grid, factor=2, random_state=rng
    )
    gsh.fit(X, y)
    print(
        "HalvingGridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (time() - start, len(gsh.cv_results_["params"]))
    )
    report(gsh.cv_results_)


if __name__ == "__main__":
    main()

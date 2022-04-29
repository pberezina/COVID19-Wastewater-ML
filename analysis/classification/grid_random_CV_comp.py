from curses import endwin
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
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent to path
    from utils import DATA_PATH

from models import CLASSIFIERS


# grab a classifier
clf = CLASSIFIERS["knn"]["model"]
param_grid = CLASSIFIERS["knn"]["param_grid"]
rng = np.random.RandomState(0)

# Utility function to report best scores
def report(results, running_results, model_type, fit_time, n_top=30):
    running_results = []
    for i in range(1, n_top + 1):
        n_top_data = {}
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            n_top_data["param"] = list(results["params"][candidate].keys())[0]
            n_top_data["value"] = list(results["params"][candidate].values())[0]
            n_top_data["mean_validation_score"] = results["mean_test_score"][candidate]
            n_top_data["std_test_score"] = results["std_test_score"][candidate]
            n_top_data["model_type"] = model_type
            n_top_data["fit_time"] = fit_time

            running_results.append(n_top_data)
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}\n".format(results["params"][candidate]))
    return running_results


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
    running_results = []

    # run randomized search
    # n_iter_search = 15
    # random_search = RandomizedSearchCV(
    #     clf, param_distributions=param_grid, n_iter=n_iter_search
    # )

    # start = time()
    # random_search.fit(X, y)
    # end = time() - start
    # print(
    #     "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    #     % ((end), n_iter_search)
    # )
    # running_results.extend(report(random_search.cv_results_, running_results, "RandomizedSearchCV", end))

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X, y)

    end = time() - start
    print(
        "GridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (end, len(grid_search.cv_results_["params"]))
    )
    running_results.extend(report(grid_search.cv_results_, running_results, "GridSearchCV", end))

    # run successive halving
    start = time()
    gsh = HalvingGridSearchCV(
        estimator=clf, param_grid=param_grid, factor=2, random_state=rng
    )
    gsh.fit(X, y)
    end = time() - start
    print(
        "HalvingGridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (end, len(gsh.cv_results_["params"]))
    )
    running_results.extend(report(gsh.cv_results_, running_results, "HalvingGridSearchCV", end))

    pd.DataFrame(running_results).to_csv(DATA_PATH.parent / "analysis/CV_tune_comp.csv")


if __name__ == "__main__":
    main()

from __future__ import annotations

import click
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

from grid_search_verification import get_param_grid_space, grid_search_CV


DATA_DIR = Path(__file__).parent.parent / "encoded_cleaned.csv"
CMAP_LIGHT = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
CMAP_BOLD = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


CLASSIFIERS = {
    "knn": dict(
        model=KNeighborsClassifier(),
        param_grid=dict(n_neighbors=get_param_grid_space(1, 30, dtype=int)),
        label="KK Nearest Neighbor",
    ),
    "lin_svc": dict(
        model=SVC(kernel="linear"),
        param_grid=dict(C=get_param_grid_space(exp=True)),
        label="Linear SVC",
    ),
    "RBF_svc": dict(
        model=SVC(),
        param_grid=dict(
            C=get_param_grid_space(exp=True), gamma=get_param_grid_space(exp=True)
        ),
        label="RBF SVC",
    ),
    "dtree": dict(
        model=DecisionTreeClassifier(),
        param_grid=dict(max_dept=get_param_grid_space(1, 10, 9, int)),
        label="Decision Tree",
    ),
}


@click.command()
@click.option(
    "-f",
    "--data_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=str(Path(__file__).parent.parent / "encoded_cleaned.csv"),
    help="Path to data file csv.",
)
def main(data_file: Path) -> None:
    df = load_data(data_file)
    x_cols = ["Confirmed Inmate Deaths", "Recovered Inmates"]
    y_col = "housing_factor"
    X = df[x_cols]
    y = df[y_col]

    scores = tune_hyper_params(X, y)
    print(scores)


def load_data(data_file: Path) -> pd.DataFrame:
    df = pd.read_csv(data_file, parse_dates=["Date"])
    return df


def tune_hyper_params(X: np.ndarray, y: np.ndarray) -> dict:
    scores = {}
    for key, classifier in CLASSIFIERS.items():
        gscv = grid_search_CV(classifier["model"], X, y, classifier["param_grid"])
        scores[key] = dict(best_params=gscv.best_params_, best_score=gscv.best_score_)
    return scores


def set_x_cols(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str = "housing_factor",
    groups: list[str] | str = "Site",
):
    df = df.dropna(subset=[*x_cols, y_col])
    X = df[x_cols]
    y = df[y_col]
    X = StandardScaler().fit_transform(X)
    groups = df[groups]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1234, stratify=groups
    )

    x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
    if X.shape[1] == 2:
        x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    else:
        x1_min, x1_max = x0_min, x0_max
    h = (
        abs(min(x0_min, x0_max, x1_min, x1_max) - max(x0_min, x0_max, x1_min, x1_max))
        / 1000
    )
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))

    return X_train, X_test, X, y_train, y_test, y, xx0, xx1


def score_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model,
) -> float:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
    return score


def plot_clasifier(
    X: np.ndarray,
    y: np.ndarray,
    xx0: np.ndarray,
    xx1: np.ndarray,
    x_cols: list[str],
    score: float,
    model,
    model_type: str,
) -> None:

    plt.figure()

    # Put the result into a color plot
    Z = model.predict(np.c_[xx0.ravel(), xx1.ravel()])
    Z = Z.reshape(xx0.shape)
    plt.pcolormesh(xx0, xx1, Z, cmap=CMAP_LIGHT, alpha=0.8, shading="auto")

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_BOLD, edgecolor="k", s=20)

    plt.xlabel(x_cols[0])
    plt.ylabel(x_cols[1])
    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())
    plt.title(f"{model_type=} with {score=}")
    plt.text(
        0.9,
        0.1,
        "{:.2f}".format(score),
        size=15,
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    plt.show()


def score_and_plot_classifiers(
    X_train, X_test, X, y_train, y_test, y, xx0, xx1, x_cols
):
    for model_type, model in CLASSIFIERS:
        print(f"Model type: {model_type}")
        score = score_model(X_train, X_test, X, y_train, y_test, model)
        plot_clasifier(X, y, xx0, xx1, x_cols, score, model, model_type)


if __name__ == "__main__":
    main()

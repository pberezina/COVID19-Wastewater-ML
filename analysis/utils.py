from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np
import scipy.stats as stats
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.base import TransformerMixin, BaseEstimator

import os

os.path.dirname(os.path.abspath(__file__))

DATA_PATH = (
    Path(os.path.dirname(os.path.abspath(__file__))).parent / "encoded_cleaned.csv"
)


def load_data(data_file: Path, date_cols=("Date",)) -> pd.DataFrame:
    df = pd.read_csv(data_file, parse_dates=list(date_cols))
    return df


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    fitted_lambda: float

    def fit(self, x: np.array) -> "BoxCoxTransformer":

        _, self.fitted_lambda = stats.boxcox(x)
        return self

    def transform(self, x: np.array) -> np.array:
        # Note that for x of length = 1 stats.boxcox will raise error
        return stats.boxcox(x, self.fitted_lambda)

    def inverse_transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        if copy:
            X = X.copy()
        return inv_boxcox(X, self.fitted_lambda)


def transform(X, transformer=RobustScaler):

    return transformer().fit_transform(X)


def inverse_transform(transformed_x, original_x, transformer=RobustScaler):
    return transformer().fit(original_x).inverse_transform(transformed_x)


def split_data(
    X: np.ndarray, y: np.ndarray, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets

    Parameters
    ----------
    X : np.ndarray
        X data
    y : np.ndarray
        y data

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1234, **kwargs
    )
    return X_train, X_test, y_train, y_test

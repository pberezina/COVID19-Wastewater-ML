from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
os.path.dirname(os.path.abspath(__file__))

DATA_PATH = (Path(os.path.dirname(os.path.abspath(__file__))).parent
             / "encoded_cleaned.csv")


def load_data(data_file: Path, date_cols=("Date",)) -> pd.DataFrame:
    df = pd.read_csv(data_file, parse_dates=list(date_cols))
    return df


def standardize(X):
    return StandardScaler().fit_transform(X)


def inverse_transform(transformed_x, original_x):
    return StandardScaler().fit(original_x).inverse_transform(transformed_x)


def split_data(
    X: np.ndarray, y: np.ndarray,
    standardize_x=True, standardize_y=True,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets

    Parameters
    ----------
    X : np.ndarray
        X data
    y : np.ndarray
        y data
    standardize_x : bool
        Standardize features by removing the mean and scaling to unit variance
    standardize_y : bool
        Standardize features by removing the mean and scaling to unit variance

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    if standardize_x:
        X = standardize(X)
    #     X = StandardScaler().fit_transform(X)
    if standardize_y:
        y = standardize(y)
    #     y = StandardScaler().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1234, **kwargs
    )
    return X_train, X_test, y_train, y_test

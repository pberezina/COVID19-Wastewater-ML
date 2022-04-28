import numpy as np
from sklearn.model_selection import GridSearchCV


def get_param_grid_space(
    seq_loq: float = 1e-4,
    seq_high: float = 1e2,
    num_in_seq: float = 100,
    dtype: np.dtype = np.float,
    exp: bool = False,
) -> np.ndarray:
    seq = np.linspace(seq_loq, seq_high, num_in_seq, dtype=dtype)
    seq = np.unique(seq)
    if exp:
        seq = np.power(10, seq)
        seq = seq - seq.min() + seq_loq
    return seq


def grid_search_CV(
    model, X: np.ndarray, y: np.ndarray, param_grid: np.array = None, **kwargs
) -> GridSearchCV:

    # use gridsearch to test all values for param_grid
    model_gscv = GridSearchCV(model, param_grid, **kwargs)
    # fit model to data
    model_gscv.fit(X, y)

    return model_gscv

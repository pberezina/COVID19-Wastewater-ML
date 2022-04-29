import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


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


CLASSIFIERS = {
    "knn": dict(
        model=KNeighborsClassifier(n_neighbors=3),
        param_grid=dict(n_neighbors=get_param_grid_space(1, 30, dtype=int)),
        label="KK Nearest Neighbor",
    ),
    # "lin_svc": dict(
    #     model=SVC(kernel="linear"),
    #     param_grid=dict(C=get_param_grid_space(exp=True)),
    #     label="Linear SVC",
    # ),
    # "RBF_svc": dict(
    #     model=SVC(),
    #     param_grid=dict(
    #         C=get_param_grid_space(exp=True), gamma=get_param_grid_space(exp=True)
    #     ),
    #     label="RBF SVC",
    # ),
    # "dtree": dict(
    #     model=DecisionTreeClassifier(),
    #     param_grid=dict(max_depth=get_param_grid_space(1, 10, 9, int)),
    #     label="Decision Tree",
    # ),
    # "rf": dict(
    #     model=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     param_grid=dict(max_depth=get_param_grid_space(1, 10, 19, int)),
    #     label="Random Forest",
    #     ),
    # "nb": dict(
    #     model=GaussianNB(),
    #     param_grid=dict(),
    #     label="Gaussian Naive Bayes",
    #     )
}

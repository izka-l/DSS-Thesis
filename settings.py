import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

MACROECONOMIC_FEATURE_NAMES = ['cci', 'cpi', 'gdp', 'rir', 'uer']



class columnDropperTransformer:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


drop_transformer = columnDropperTransformer(MACROECONOMIC_FEATURE_NAMES)
param_grid = [
    {
        "clf": [SVC()],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": [1, 0.1, 0.01, 0.001],
        "clf__kernel": ["rbf", "linear", "poly", "sigmoid"],
        "marcro_drop": ["passthrough", drop_transformer],
    },
    {
        "clf": [LogisticRegression(max_iter=500)],
        "clf__penalty": ["l1", "l2"],
        "clf__C": np.logspace(-3, 1, 5),
        "clf__solver": ["liblinear"],
        "marcro_drop": ["passthrough", drop_transformer],
    },
    {
        "clf": [LogisticRegression(max_iter=500)],
        "clf__penalty": ["l2"],
        "clf__C": np.logspace(-3, 1, 5),
        "clf__solver": ["lbfgs", "newton-cg"],
        "marcro_drop": ["passthrough", drop_transformer],
    },
    {
        "clf": [GaussianNB()],
        "clf__var_smoothing": np.logspace(-4, 0, num=5),
        "marcro_drop": ["passthrough", drop_transformer]
    },
    {
        "clf": [MLPClassifier()],
        "clf__hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
        "clf__activation": ["tanh", "relu"],
        "clf__solver": ["sgd", "adam"],
        "clf__alpha": [0.001, 0.01],
        "clf__learning_rate": ["constant", "adaptive"],
        "marcro_drop": ["passthrough", drop_transformer],
    },
    {
        "clf": [RandomForestClassifier()],
        "clf__max_depth": [10, 12, 14, 16, 18, 20],
        "clf__min_samples_leaf": [2, 5, 10, 15, 20, 25],
        "clf__bootstrap": [True, False],
        "marcro_drop": ["passthrough", drop_transformer],
    },
]

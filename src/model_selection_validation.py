import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from scipy.stats import randint
# REMOVED MODULE IMPORT STATEMENTS


# Modified example work script of a machine learning Python data pipeline.
# This script is intended as a work example of Adam Morphy's work with the Vancouver Whitecaps FC, and has been modified outside its original data pipeline.
# Data names and references have been removed.

#################################################
#                   _______
#                  |       |              
#            o     |       |
#          -()-   o
#           |\
#################################################


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def super_split(df):
    # split the data to train and test df
    # return X_train, y_train, X_test, y_test

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)

    X_train, y_train = (
        train_df.drop(columns=["COL1"]),
        train_df["COL1"],
    )
    X_test, y_test = (
        test_df.drop(columns=["COL1"]),
        test_df["COL1"],
    )
    return X_train, y_train, X_test, y_test


def super_features():

    drop_features = ["COL1"]
    numeric_features = [
        "COL2",
        "COL3",
        "COL4",
        "COL5",
        "COL6",
        "COL7",
        "COL8",
    ]
    binary_features = ["COL8"]
    categorical_features = ["COL9"]
    target = "COL10"

    return (
        drop_features,
        numeric_features,
        binary_features,
        categorical_features,
        target,
    )


def preprocessor(
    drop_features, numeric_features, binary_features, categorical_features
):
    """
    Returns processed data using column transformer
    Parameters
    ----------
    numeric_features : list of numeric features of the DataFrame
        numeric features which are to be scaled
    categorical_features : list of categorical features of the DataFrame
        categorical features which will be one hot encoded
    Returns
    ----------
        column transformer of the features
    """

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop="if_binary"), binary_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    )
    return preprocessor


def super_multiple_models(preprocessor, X_train, y_train):
    """
    Returns DataFrame with validation scores of classification models
    Parameters
    ----------
    preprocessor : column transformer of the DataFrame
        column transformer with scaling and OHE on features
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data
    Returns
    ----------
        DataFrame of the validation scores of different models
    """

    ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]
    models = {
        "RBF SVM": SVC(),
        "random forest": RandomForestClassifier(
            class_weight="balanced", random_state=2
        ),
        "xgboost": XGBClassifier(scale_pos_weight=ratio, random_state=2),
        "lgbm": LGBMClassifier(scale_pos_weight=ratio, random_state=2),
    }

    scoring_metric = make_scorer(
        f1_score, average="macro"
    )  # f1_score chosen as there is a class imbalance, average="macro"
    results = {}

    # Calculating mean cross validation score
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        results[name] = mean_std_cross_val_scores(
            pipe, X_train, y_train, return_train_score=True, scoring=scoring_metric
        )

    return pd.DataFrame(results).T


def super_hyperparameter_tuning(preprocessor, X_train, y_train):
    """
    Returns the best parameter values and LGBM with those values
    Parameters
    ----------
    preprocessor : column transformer of the DataFrame
        column transformer with scaling and OHE on features
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data
    Returns
    ----------
        Returns the best LGBM model with its tuned hyperparameters
    """

    ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]

    models = {
        "lgbm": LGBMClassifier(scale_pos_weight=ratio, random_state=2),
    }

    scoring_metric = make_scorer(f1_score, average="macro")

    param_grid_lgbm = {
        "lgbmclassifier__n_estimators": randint(10, 100),
        # "lgbmclassifier__max_depth": randint(low=2, high=20),
        "lgbmclassifier__learning_rate": [0.01, 0.1],
        "lgbmclassifier__subsample": [0.5, 0.75, 1],
    }

    pipe_lgbm = make_pipeline(
        preprocessor,
        models["lgbm"],
    )

    random_search_lgbm = RandomizedSearchCV(
        pipe_lgbm,
        param_grid_lgbm,
        n_iter=50,
        verbose=1,
        n_jobs=1,
        scoring=scoring_metric,
        random_state=123,
        return_train_score=True,
    )

    random_search_lgbm.fit(X_train, y_train)

    best_lgbm_model = random_search_lgbm.best_estimator_

    results = {}
    results["lgbm (tuned)"] = mean_std_cross_val_scores(
        best_lgbm_model,
        X_train,
        y_train,
        return_train_score=True,
        scoring=scoring_metric,
    )

    table = pd.DataFrame(results).T.rename(
        columns={"test_score": "Validation f1 Score", "train_score": "Train f1 Score"}
    )

    return best_lgbm_model, table

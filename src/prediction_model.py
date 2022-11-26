# Author: Vikram Grewal
# Date: 2022-11-25

"""Using the processed data, transforms the data for model fitting, fits different models for comparison,
   selects the best model, outputs comparison table between models, and outputs plot of compairson between
   actual and predicted values from the model.
   
Usage: src/prediction_model.py --training_file=<training_file> --testing_file=<testing_file> --results_dir=<results_dir> 
 
Options:
--training_file=<training_file>     Path (including filename) of where to locally read in the training data file
--testing_file=<testing_file>       Path (including filename) of where to locally read in the testing data file
--results_dir=<results_dir>         Path to directory of where to locally write the results
"""

# Example:
# python src/prediction_model.py --training_file="data/processed/training_split.csv" --testing_file="data/processed/testing_split.csv" --results_dir="results"

# Imports -----------------------------------------------------------------------------------------
import os
from docopt import docopt
import pandas as pd
import numpy as np
import altair as alt
from ast import literal_eval

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from sklearn.base import TransformerMixin

import warnings
warnings.filterwarnings('ignore')
# -------------------------------------------------------------------------------------------------

opt = docopt(__doc__)
alt.renderers.enable('mimetype')
alt.data_transformers.enable('data_server')

class MyMultiLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit(self, x, y=None):
        self.encoder.fit(x)
        self.classes_ = self.encoder.classes_
        return self

    def transform(self, x, y=None):
        return self.encoder.transform(x)
    
    def get_params(self, deep=True):
        return self.encoder.get_params()

def main(training_file, testing_file, results_dir):
    # Create the training and testing data frames
    train_df = pd.read_csv("../data/processed/training_split.csv")
    test_df = pd.read_csv("../data/processed/testing_split.csv")

    # Turn the list-like string columns into actual lists for MultiLabelBinarizer
    categorical_list_features = ["boardgamecategory",
                                 "boardgamemechanic", 
                                 "boardgamefamily", 
                                 "boardgamedesigner", 
                                 "boardgameartist", 
                                 "boardgamepublisher"]
    for feat in categorical_list_features:
        train_df[feat] = train_df[feat].apply(literal_eval)
        test_df[feat] = test_df[feat].apply(literal_eval)

    # Split the data into respective x and y dataframes
    X_train, y_train = train_df.drop(columns="average"), train_df["average"]
    X_test, y_test = test_df.drop(columns="average"), test_df["average"]

    # Defining features for preprocessor
    numerical_features = ["yearpublished", 
                          "minplayers", 
                          "maxplayers", 
                          "playingtime", 
                          "minplaytime", 
                          "maxplaytime", 
                          "minage"]

    text_feature = "description"

    # Define scoring methods for model evaluation
    scoring_dict = {
    "r2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mse": "neg_mean_squared_error",
    }   

    # Define dictionary for storing cross-validation results
    cross_val_results = {}

    # Define preprocessor for column transformations
    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_features),
        (CountVectorizer(stop_words="english", max_features=1000), text_feature),
        (MyMultiLabelBinarizer(), "boardgamecategory"),
        (MyMultiLabelBinarizer(), "boardgamemechanic"),
        (MyMultiLabelBinarizer(), "boardgamefamily"),
        (MyMultiLabelBinarizer(), "boardgamedesigner"),
        (MyMultiLabelBinarizer(), "boardgameartist"),
        (MyMultiLabelBinarizer(), "boardgamepublisher")
    )

    # Create and score dummy regressor for baseline model
    dummy_regressor = DummyRegressor()
    cross_val_results['dummy_regressor'] = pd.DataFrame(cross_validate(dummy_regressor, X_train, y_train, return_train_score = True, scoring=scoring_dict)).agg(['mean', 'std']).round(3).T    cross_val_results['dummy_regressor']

    # Create pipeline for Ridge model hyperparameter optimization
    pipe_ridge = make_pipeline(
        preprocessor,
        Ridge()
    )

    # Create parameter distribution for Ridge alpha hyperparameter
    param_dist_ridge = {
        "ridge__alpha": 10.0 ** np.arange(-6, 6, 2)
    }

    # Perform randomized search for optimal hyperparameter
    ridge_search = RandomizedSearchCV(
        pipe_ridge, param_dist_ridge, n_iter=12, n_jobs=-1, return_train_score=True
    )
    ridge_search.fit(X_train, y_train)

    # Define new Ridge pipeline with optimal alpha from search
    pipe_ridge_opt = make_pipeline(
        preprocessor,
        Ridge(alpha=ridge_search.best_params_["ridge__alpha"])
    )

    # Perform cross-validation and store results in data frame
    cross_val_results["ridge"] = pd.DataFrame(cross_validate(pipe_ridge_opt,
                                                             X_train, 
                                                             y_train, 
                                                             return_train_score=True, 
                                                             scoring=scoring_dict)).agg(['mean', 'std']).round(3).T

    # Define Random Forest Regressor (RFR) pipeline
    pipe_rfr = make_pipeline(
        preprocessor,
        RandomForestRegressor(n_jobs=-1)
    )

    # Define RFR hyperparameter options for optimization
    param_dist_rfr = {
        "randomforestregressor__max_depth": np.arange(20, 100, 2),
        "randomforestregressor__bootstrap": [True, False],
        "randomforestregressor__min_samples_leaf": [1, 2, 4],
        "randomforestregressor__min_samples_split": [2, 5, 10]
    }

    # Perform hyperparameter optimization for RFR
    rfr_search = RandomizedSearchCV(
        pipe_rfr, param_dist_rfr, n_iter=10, n_jobs=-1, return_train_score=True
    )
    rfr_search.fit(X_train.iloc[0:2000], y_train.iloc[0:2000])

    # Define new pipeline for RFR using optimized hyperparameters
    pipe_rfr_opt = make_pipeline(
        preprocessor,
        RandomForestRegressor(max_depth=rfr_search.best_params_["randomforestregressor__max_depth"],
                            bootstrap=rfr_search.best_params_["randomforestregressor__bootstrap"],
                            min_samples_leaf=rfr_search.best_params_["randomforestregressor__min_samples_leaf"],
                            min_samples_split=rfr_search.best_params_["randomforestregressor__min_samples_split"])
    )

    # Perform cross-validation and store results in data frame
    cross_val_results["random_forest"] = pd.DataFrame(cross_validate(pipe_rfr_opt, X_train.iloc[0:2000], y_train.iloc[0:2000], return_train_score=True, scoring=scoring_dict)).agg(['mean', 'std']).round(3).T

    # Drop uneeded columns from scoring dataframes (originally kept for analysis purposes)
    cross_val_results["dummy_regressor"] = cross_val_results["dummy_regressor"].drop(columns="std").rename(columns={"mean": "Dummy_Regressor"})
    cross_val_results["random_forest"] = cross_val_results["random_forest"].drop(columns="std").rename(columns={"mean": "Random_Forest"})
    cross_val_results["ridge"] = cross_val_results["ridge"].drop(columns="std").rename(columns={"mean": "Ridge"})

    # Combine all results into one results data frame
    cross_val_results_df = cross_val_results["dummy_regressor"].join(cross_val_results["ridge"], how="inner").join(cross_val_results["random_forest"], how="inner")

    # Fit the optimal RFR model
    pipe_rfr_opt.fit(X_train, y_train)

    # Create dataframe for actual scores from the testing data and the predicted scores from the model
    results_dict = {
        "Actual Scores": y_test,
        "Predicted Scores": pipe_rfr_opt.predict(X_test)
    }
    results_df = pd.DataFrame(results_dict)

    # Create a dataframe for plotting a 1:1 line for visualizing what 100$ accuracy would look like
    line_df = pd.DataFrame(
        {
            "Actual Scores": [2, 10],
            "Predicted Scores": [2, 10]
        }
    )
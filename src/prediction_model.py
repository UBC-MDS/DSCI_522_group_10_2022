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


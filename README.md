# Board Game User Rating Predictor

- contributors: Marian Agyby, Ashwin Babu, Vikram Grewal, Eric Tsai
- URL of the project repo is [here](https://github.com/UBC-MDS/boardgame_rating_predictor)

## Project Proposal

In this project we aim to answer the following question: Given certain characteristics about a new board game, how would users rate the board game? Answering this question will help board game creators understand which characteristics enhance user enjoyment and improve their developing capabilities towards a successful game, minimizing their R&D time and developing a popular new board game.

To answer this question, we are using a large data set containing user ratings and reviews for thousands of board games, created by [BoardGameGeek](https://boardgamegeek.com/) and made available by [tidytuesday](https://github.com/rfordatascience/tidytuesday), which can be found [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25). The data consists of two data sets, one containing the user ratings, and the other containing information about the board games, including names and descriptions, as well as several characteristics such as playing time, minimum age, number of  players, etc. We will join the two data sets and build a regression model that predicts the average user rating based on various game features.

First we will split the data into 60% training set and 40% test set, then perform exploratory data analysis on the training set to assess which features are the most appropriate to train the model. One table will summarize the feature data types and number of missing values, and another will display the summary statistics. These tables will allow us to determine how to clean up and filter the data. A distribution of the average rating target variable will be displayed as a histogram, and used to assess whether the data is imbalanced or skewed. Distributions of the features will be displayed as histograms to show the most common feature values. Additionally, correlations between features will be displayed in a table and plotted as pairwise comparisons to identify features that are strongly correlated, and thus, identifying repetitive features that can be dropped from the training process.

Since the target we are trying to predict is continuous, and the features include a mixture of categorical and continuous variables, we will test out a few predictive regression models and assess their performance, then select the one that performs with the highest accuracy as the final model. The data set is large, has many features, and will require scaling transformations, so a few suitable models that we will test out are the `Ridge()`, `SVR()`, and `RandomForestRegressor()` models. We will also use `RandomSearchCV()` to cross-validate and optimize the models' hyperparameter values. Once the final model is selected and fitted to the entire training set, we will use it to predict average user ratings on the test set, measure the accuracy of the model, and report the model's performance results in a table.

The exploratory data analysis report can be found [here](https://github.com/UBC-MDS/DSCI_522_group_10_2022/blob/main/src/boardgame_rating_eda.ipynb).

## Usage

To reproduce this analysis, clone this repository, install the dependencies listed below, and run the following two commands at the command line/terminal from the root directory of this project:


    python src/download_data.py --url=https://github.com/rfordatascience/tidytuesday/blob/master/data/2022/2022-01-25/ratings.csv --out_file=data/raw/ratings.csv


    python src/download_data.py --url=https://github.com/rfordatascience/tidytuesday/blob/master/data/2022/2022-01-25/details.csv --out_file=data/raw/details.csv


## Dependencies

- Python 3.10.6 and Python packages:
    - docopt-ng==0.8.1
    - requests 2.27.1
    - numpy==1.23.4
    - pandas==1.4.4
    - altair==4.2.0
    - altair_saver
    - scikit-learn==1.1.3
    - ipykernel

## License

All Board Game User Rating Predictor materials are licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License and the MIT License.

# References

BoardGameGeek, LLC. 2022. "Board Games". Retrieved November 16, 2022 from github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25.
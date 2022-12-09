# Board Game User Rating Predictor

- contributors: Marian Agyby, Ashwin Babu, Vikram Grewal, Eric Tsai
- URL of the project repo is [here](https://github.com/UBC-MDS/boardgame_rating_predictor)

## Project Proposal

In this project we aim to answer the following question: Given certain characteristics about a new board game, how would users rate the board game? Answering this question will help board game creators understand which characteristics enhance user enjoyment and improve their developing capabilities towards a successful game, minimizing their R&D time and developing a popular new board game.

To answer this question, we are using a large data set containing user ratings and reviews for thousands of board games, created by [BoardGameGeek](https://boardgamegeek.com/) and made available by [tidytuesday](https://github.com/rfordatascience/tidytuesday), which can be found [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25). The data consists of two data sets, one containing the user ratings, and the other containing information about the board games, including names and descriptions, as well as several characteristics such as playing time, minimum age, number of  players, etc. We have merged the two data sets and built multiple regression models that predict the average user rating based on various features.

## Analysis

First we will split the data into 50% training set and 50% test set (because of the time it takes to train the model), then perform exploratory data analysis on the training set to assess which features are the most appropriate to train the model. One table will summarize the feature data types and number of missing values, and another will display the summary statistics. These tables will allow us to determine how to clean up and filter the data. A distribution of the average rating target variable will be displayed as a [histogram](https://github.com/UBC-MDS/boardgame_rating_predictor/blob/main/results/rating_distribution.png), and will be used to assess whether the data is imbalanced or skewed. Distributions of the numeric features will be displayed as [histograms](https://github.com/UBC-MDS/boardgame_rating_predictor/blob/main/results/numeric_feature_distribution.png) to show the most common numeric feature values. Additionally, correlations between features will be displayed in a table and plotted as pairwise comparisons to identify features that are strongly correlated, and thus, identifying repetitive features that can be dropped from the training process.

Since the target we are trying to predict is continuous, and the features include a mixture of categorical and continuous variables, we will test out a few predictive regression models and assess their performance, then select the one that performs with the highest accuracy as the final model. The data set is large, has many features, and will require scaling transformations, so a few suitable models that we will test out are the `Ridge()`, and `RandomForestRegressor()` models. We will also use `RandomSearchCV()` to cross-validate and optimize the models' hyperparameter values. Once the final model is selected and fitted to the entire training set, we will use it to predict average user ratings on the test set, measure the accuracy of the model, and report the model's performance results in a table.

The exploratory data analysis report can be found [here](https://github.com/UBC-MDS/DSCI_522_group_10_2022/blob/main/src/boardgame_rating_eda.ipynb).


## Report

The final report of the project can be found [here](https://github.com/UBC-MDS/boardgame_rating_predictor/tree/main/doc)


## Usage
To replicate this analysis, you can follow these instructions and run the corresponding make commands. These instructions will need to be ran in a Unix shell.
  1. Download the dependency file from the .yaml [file](https://github.com/UBC-MDS/boardgame_rating_predictor/blob/main/envboard.yaml)
  
  2. Create and activate the environment
  
    conda env create -f envboard.yaml
    conda activate envboard
  
  3. Clone the repository from:
  
  
      https://github.com/UBC-MDS/boardgame_rating_predictor.git
  
  4. Move to the cloned directory
  
  
    cd boardgame_rating_predictor
  
  5. To replicate the analysis in its entirety, you can run the following command.

    make all

  6. To delete the files and figures created from the analysis and return the repository to a clean state, run the following.

    make clean
  
  7. To run individual parts of the script, follow the instructions below:
    
    To Download the raw data set use the command:
  
      make data/raw/ratings.csv
      make data/raw/details.csv
    
    To perform preprocessing on the raw data, use the command:

      make data/processed
    
    To just run the exploratory data analysis and generate the figures, use the command:

      make eda
    
    To create the regression model and generate the scores from prediction on the data, use the command:

      make predict_model
    
    To generate the final report, use the command:

      make doc/boardgame_rating_predictor_report.html

  8. For a more in-depth look at the exploratory data analysis, see [link](https://github.com/UBC-MDS/boardgame_rating_predictor/blob/main/src/boardgame_rating_eda.ipynb) or run the file on any IDE.
  
    
  9. If you want to check the model performance comparison click [here](https://github.com/UBC-MDS/boardgame_rating_predictor/blob/main/results/model_comparison_table.csv)


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
    - matplotlib>=3.2.2
    - requests>=2.24.0
    - graphviz
    - python-graphviz
    - eli5
    - shap
    - jinja2
    - selenium<4.3.0
    - imbalanced-learn
    - pip
    - lightgbm
    - vl_convert

## Makefile dependency diagram
<img src="Makefile.png">

## License

All Board Game User Rating Predictor materials are licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License and the MIT License.

# References

BoardGameGeek, LLC. 2022. "Board Games". Retrieved November 16, 2022 from github.com/rfordatascience/tidytuesday/tree/master/data/2022/2022-01-25.
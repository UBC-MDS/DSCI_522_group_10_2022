"""
A script reads in a processed training dataset and performs EDA to output figures as png files.

Usage: src/eda_boardgame.py --in_file=<in_file> --out_dir=<out_dir> 
 
Options:
--in_file=<in_file>     Path (including filename) of where to locally read in the file
--out_dir=<out_dir>     Path to directory of where to locally write the output figures
"""

# Example:
# python src/eda_boardgame.py --in_file="data/processed/training_split.csv" --out_dir="results"

# input file: "data/processed/training_split.csv"
# output directory: "results"


import os
import pandas as pd
import numpy as np
import altair as alt
import vl_convert as vlc
from ast import literal_eval
from docopt import docopt
from sklearn.preprocessing import MultiLabelBinarizer

alt.data_transformers.disable_max_rows()
opt = docopt(__doc__)

def main(in_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_df = pd.read_csv(in_file)
    # process columns that appear like lists but are actually strings
    # reference: https://stackoverflow.com/questions/25572247/how-to-convert-array-string-to-an-array-in-python
    categorical_features = ["boardgamecategory", "boardgamemechanic", "boardgamefamily", "boardgameartist", "boardgamepublisher"]
    for feat in categorical_features:
        train_df[feat] = train_df[feat].apply(literal_eval)
    binarized_rating_df = augment_df(train_df)  # augment training set to include a new column "rating" for easier visualization

    rating_plot = plot_rating_distribution(train_df)
    numeric_feats_bar_plot = plot_numeric_feature_distribution(train_df)
    top_10_categories_plot = plot_top_10_categories(binarized_rating_df)
    save_chart(rating_plot, out_dir + "/rating_distribution.png")
    save_chart(numeric_feats_bar_plot, out_dir + "/numeric_feature_distribution.png")
    save_chart(top_10_categories_plot, out_dir + "/top_10_categories.png")


def plot_rating_distribution(df):
    '''
    Creates histogram for distribution of average rating
    '''
    rating_plot = alt.Chart(
        df,
        title = "Distribution of Average Rating"
    ).mark_bar().encode(
        x = alt.X("average", bin=alt.Bin(maxbins=30), title="Average Rating"),
        y = alt.Y("count()", title="Count")
    )
    return rating_plot


def plot_numeric_feature_distribution(df):
    '''
    Creates repeated chart of distribution of numeric features
    '''
    numeric_feats = df.select_dtypes(include="number").columns.tolist()[0:-1]
    numeric_feats_bar_chart = alt.Chart(df).mark_bar().encode(
        x = alt.X(alt.repeat(), type="quantitative"),
        y = "count()"
    ).properties(
        width=200,
        height=200
    ).repeat(
        numeric_feats,
        columns=3
    )
    numeric_feats_bar_chart = numeric_feats_bar_chart.properties(
        title = alt.TitleParams(text="Distribution of Numeric Features", anchor="middle")
    )
    return numeric_feats_bar_chart


def augment_df(df):
    '''
    Adds a new column with value of either "high" or "low to the dataframe based on the average column's value
    '''
    def binarize_rating(row):
        if row["average"] >= 7:
            rating = "high"
        else:
            rating = "low"
        return rating

    binarized_rating_df = df.copy()
    binarized_rating_df["rating"] = binarized_rating_df.apply(binarize_rating, axis=1)
    return binarized_rating_df


def plot_top_10_categories(df):
    '''
    Creates a bar chart of the top 10 boardgame categories
    '''
    mlb = MultiLabelBinarizer()
    category_trans = mlb.fit_transform(df["boardgamecategory"])
    category_binary_df = pd.concat([pd.DataFrame(category_trans, columns=mlb.classes_), df["rating"].reset_index()], axis=1)
    top_10_categories = category_binary_df.sum(numeric_only=True).sort_values(ascending=False)[1:11].index.tolist()
    # Group the categories by rating to get the count of each category for each rating
    category_grouped = category_binary_df.groupby("rating").sum().T
    category_grouped["category"] = category_grouped.index
    category_grouped.reset_index(drop=True)
    category_long = category_grouped.melt("category")
    top_10_categories_plot = alt.Chart(
        category_long.query("category in @top_10_categories"),
        title = "Top 10 Categories and Their Rating Comparisons"
    ).mark_bar().encode(
        x = alt.X("rating", axis=alt.Axis(title=None, labels=False, ticks=False)),
        y = alt.Y("value", title="Count"),
        column = alt.Column('category', header=alt.Header(title=None, labelOrient='bottom')),
        color = alt.Color("rating", title="Rating")
    )
    return top_10_categories_plot


# Credits to Joel Ostblom for this function
def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")   


if __name__ == "__main__":
    main(opt["--in_file"], opt["--out_dir"])


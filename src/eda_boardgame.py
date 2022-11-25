# Author: Eric Tsai
# Date: 2022-11-24 

"""
A script reads in a processed training dataset and performs EDA to output 5 figure
    - rating distribution
    - numeric feature distribution
    - top 10 boardgame categories
    - top 10 boardgame mechanics
    - top 10 boardgame families
as png files.

Usage: src/eda_boardgame.py --in_file=<in_file> --out_dir=<out_dir> 
 
Options:
--in_file=<in_file>     Path (including filename) of where to locally read in the file
--out_dir=<out_dir>     Path to directory of where to locally write the output figures
"""

# Example:
# python src/eda_boardgame.py --in_file="data/processed/training_split.csv" --out_dir="results/"

# input file: "data/processed/training_split.csv"
# output directory: "results/"


import os
import pandas as pd
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
    top_10_mechanics_plot = plot_top_10_mechanics(binarized_rating_df)
    top_10_families_plot = plot_top_10_families(binarized_rating_df)
    save_chart(rating_plot, out_dir + "rating_distribution.png")
    save_chart(numeric_feats_bar_plot, out_dir + "numeric_feature_distribution.png")
    save_chart(top_10_categories_plot, out_dir + "top_10_boardgame_categories.png")
    save_chart(top_10_mechanics_plot, out_dir + "top_10_boardgame_mechanics.png")
    save_chart(top_10_families_plot, out_dir + "top_10_boardgame_families.png")

    # sanity check unit tests
    test_figure_images_exist(out_dir + "rating_distribution.png")
    test_figure_images_exist(out_dir + "numeric_feature_distribution.png")
    test_figure_images_exist(out_dir + "top_10_boardgame_categories.png")
    test_figure_images_exist(out_dir + "top_10_boardgame_mechanics.png")
    test_figure_images_exist(out_dir + "top_10_boardgame_families.png")


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


def plot_top_10_categories(df):
    '''
    Creates a bar chart of the top 10 boardgame categories
    '''
    category_binary_df = binarize_list_column(df, "boardgamecategory")
    top_10_categories = category_binary_df.sum(numeric_only=True).sort_values(ascending=False)[1:11].index.tolist()
    category_long = df_to_long_format(category_binary_df, "boardgamecategory")
    top_10_categories_plot = alt.Chart(
        category_long.query("boardgamecategory in @top_10_categories"),
        title = "Comparing ratings of top 10 boardgame categories"
    ).mark_bar().encode(
        x = alt.X("rating", axis=alt.Axis(title=None, labels=False, ticks=False)),
        y = alt.Y("value", title="Count"),
        column = alt.Column('boardgamecategory', header=alt.Header(title=None, labelOrient='bottom'), sort=top_10_categories),
        color = alt.Color("rating", title="Rating")
    )
    return top_10_categories_plot


def plot_top_10_mechanics(df):
    '''
    Creates a bar chart of the top 10 boardgame mechanics
    '''
    mechanic_count_df = binarize_list_column(df, "boardgamemechanic")
    top_10_mechanics = mechanic_count_df.sum(numeric_only=True).sort_values(ascending=False)[1:11].index.tolist()
    mechanic_long = df_to_long_format(mechanic_count_df, "boardgamemechanic")
    top_10_mechanic_plot = alt.Chart(
        mechanic_long.query("boardgamemechanic in @top_10_mechanics"),
        title = "Comparing ratings of top 10 boardgame mechanics"
    ).mark_bar().encode(
        alt.X("rating", axis=alt.Axis(title=None, labels=False, ticks=False)),
        alt.Y("value", title="Count"),
        column = alt.Column(
            'boardgamemechanic', 
            header = alt.Header(title=None, labelOrient='bottom', labelAngle=330, labelAnchor="end"),
            sort = top_10_mechanics
        ),
        color = alt.Color("rating", title="Rating")
    )
    return top_10_mechanic_plot


def plot_top_10_families(df):
    '''
    Creates a bar chart of the top 10 boardgame families
    '''
    family_count_df = binarize_list_column(df, "boardgamefamily")
    top_10_families = family_count_df.sum(numeric_only=True).sort_values(ascending=False)[1:11].index.tolist()
    family_long = df_to_long_format(family_count_df, "boardgamefamily")
    top_10_family_plot = alt.Chart(
        family_long.query("boardgamefamily in @top_10_families"),
        title = "Comparing ratings of top 10 boardgame mechanics"
    ).mark_bar().encode(
        alt.X("rating", axis=alt.Axis(title=None, labels=False, ticks=False)),
        alt.Y("value", title="Count"),
        column = alt.Column(
            'boardgamefamily', 
            header = alt.Header(title=None, labelOrient='bottom', labelAngle=330, labelAnchor="end"),
            sort = top_10_families
        ),
        color = alt.Color("rating", title="Rating")
    )
    return top_10_family_plot


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


def binarize_list_column(df, column):
    '''
    Helper function for plotting categorical features.
    Transform the input column containing list objects in the DataFrame into separate binary columns and concatenates the 
    rating column to the transformed DataFrame
    '''
    mlb = MultiLabelBinarizer()
    transformed = mlb.fit_transform(df[column])
    return pd.concat([pd.DataFrame(transformed, columns=mlb.classes_), df["rating"].reset_index()], axis=1)


def df_to_long_format(df, column):
    '''
    Helper function for plotting categorical features. 
    Group the DataFrame by rating and transform on the input column into a long format
    '''
    grouped_df = df.groupby("rating").sum().T
    grouped_df[column] = grouped_df.index
    grouped_df.reset_index(drop=True)
    return grouped_df.melt(column)


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


def test_figure_images_exist(file_path):
    assert os.path.isfile(file_path), f"Could not locate {file_path}."


if __name__ == "__main__":
    main(opt["--in_file"], opt["--out_dir"])

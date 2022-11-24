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

    # plot distribution of rating
    rating_plot = alt.Chart(
        train_df,
        title = "Distribution of Average Rating"
    ).mark_bar().encode(
        x = alt.X("average", bin=alt.Bin(maxbins=30), title="Average Rating"),
        y = alt.Y("count()", title="Count")
    )
    # plot distribution of numeric features
    numeric_feats = train_df.select_dtypes(include="number").columns.tolist()[0:-1]
    numeric_feats_bar_chart = alt.Chart(train_df).mark_bar().encode(
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
    # plot correlation matrix
    corr_matrix = train_df.corr('spearman').style.background_gradient()

    save_chart(rating_plot, out_dir + "/rating_distribution.png")
    save_chart(numeric_feats_bar_chart, out_dir + "/numeric_feature_distribution.png")


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


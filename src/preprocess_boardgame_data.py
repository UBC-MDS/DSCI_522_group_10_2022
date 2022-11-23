# Author: Eric Tsai
# Date: 2022-11-23  

"""Given the two raw csv files on boardgame rating and information, cleans up the file, splits the data into training
   and testing portions and writes the data to separate files

Usage: src/preprocess_boardgame_data.py --in_file1=<in_file1> --in_file2=<in_file2> --out_dir=<out_dir> 
 
Options:
--in_file1=<in_file1>   Path (including filename) of where to locally read in the file
--in_file2=<in_file2>   Path (including filename) of where to locally read in the file
--out_dir=<out_dir>     Path to directory of where to locally write the processed data
"""

# Example:
# python src/preprocess_boardgame_data.py --in_file1="data/raw/ratings.csv" --in_file2="data/raw/details.csv" --out_dir="data/processed"

# input file 1: "data/raw/ratings.csv"
# input file 2: "data/raw/details.csv"
# output directory: "data/processed"


import os
import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(in_file1, in_file2, out_dir):
    ratings = pd.read_csv(in_file1, header=1)
    details = pd.read_csv(in_file2, header=1)
    ratings = ratings[["id", "average"]]
    details = details.drop(columns=["num", "primary", "owned", "trading", "wanting", "wishing"])
    boardgame_df = details.merge(ratings, on="id", how="left").drop(columns=["id"])
    boardgame_df = boardgame_df.drop(columns=["boardgameexpansion", "boardgameimplementation"])
    boardgame_df = boardgame_df.dropna(subset=["description", "boardgamefamily", "boardgameartist", "boardgamepublisher"])  # remove rows that contain NAs for these columns
    train_df, test_df = train_test_split(boardgame_df, test_size=0.5, random_state=42)

    # write the training split to the output directory 
    try:
        train_df.to_csv(out_dir + "/training_split.csv", index=False)
    except:
        os.makedirs(os.path.dirname(out_dir + "/training_split.csv"))
        train_df.to_csv(out_dir + "/training_split.csv", index=False)
    # write the testing split to the output directory
    try:
        test_df.to_csv(out_dir + "/testing_split.csv", index=False)
    except:
        os.makedirs(os.path.dirname(out_dir + "/testing_split.csv"))
        test_df.to_csv(out_dir + "/testing_split.csv", index=False)
    
    # Sanity Check Unit Tests
    test_merged_df_columns_correct(boardgame_df)
    test_training_split_file_exist()
    test_testing_split_file_exist()


def test_merged_df_columns_correct(df):
    assert len(df.columns) == 15, f"Expected 15 columns in the merged dataframe, got {len(df.columns)}."

def test_training_split_file_exist():
    file_path = "data/processed/training_split.csv"
    assert os.path.isfile(file_path), "Could not find training_split.csv in the preprocessed data folder."

def test_testing_split_file_exist():
    file_path = "data/processed/testing_split.csv"
    assert os.path.isfile(file_path), "Could not find testing_split.csv in the preprocessed data folder."


if __name__ == "__main__":
    main(opt["--in_file1"], opt["--in_file2"], opt["--out_dir"])
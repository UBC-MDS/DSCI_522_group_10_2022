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
    train_df, test_df = train_test_split(boardgame_df, test_size=0.5, random_state=42)

    try:
        train_df.to_csv(out_dir + "/training_split.csv", index=False)
        print("success 1")
    except:
        os.makedirs(os.path.dirname(out_dir + "/training_split.csv"))
        train_df.to_csv(out_dir + "/training_split.csv", index=False)
    try:
        test_df.to_csv(out_dir + "/testing_split.csv", index=False)
        print("success 2")
    except:
        os.makedirs(os.path.dirname(out_dir + "/testing_split.csv"))
        test_df.to_csv(out_dir + "/testing_split.csv", index=False)
        

if __name__ == "__main__":
    main(opt["--in_file1"], opt["--in_file2"], opt["--out_dir"])
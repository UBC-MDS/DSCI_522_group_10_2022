# Author: Ashwin Babu
# Date: 2022-11-18

"""
A script that downloads data in csv format from the web to a local filepath as a csv.

Usage: src/download_data.py --url=<url> --out_file=<out_file> 
 
Options:
--url=<url>             URL from where to download the data (must be in standard csv format)
--out_file=<out_file>   Specify the path including filename of where to locally write the file
"""

# Example:
# python src/download_data.py --url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv" --out_file="data/raw/details.csv"
# python src/download_data.py --url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv" --out_file="data/raw/ratings.csv"

import os
import pandas as pd
from docopt import docopt
import requests

opt = docopt(__doc__) # This would parse into dictionary in python

# deatils url: url_file= "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv"
# ratings url: url_file = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv"
# details output file: out_red = "data/raw/details.csv"
# ratings output file: out_white = "data/raw/ratings.csv"

def main(url, out_file):
    """
    Download the data from the given url and save it as a csv file with path and file name as mentioned in out_file.

    Parameters:
    url (str): The raw url of the dataset you want to download
    out_file (str):  Path (including the filename) of where to write the file with downloaded data locally

    Returns:
    Stores the csv file in the location and given name in the out_file
    Example:
    main("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv", "data/raw/red_wine.csv")
    """
    # A try catch block to check if the url given exist and data is retrievable from it 
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as req:
        print(req)
        print("Website at the provided url does not exist")


    data = pd.read_csv(url, header=None) # reading the data in a pandas dataframe

    # A try catch block to save the data as a csv file to the targetted path.
    try:
        data.to_csv(out_file, index=False)
    except:
        os.makedirs(os.path.dirname(out_file))
        data.to_csv(out_file, index=False)


if __name__ == "__main__":
    main(opt["--url"], opt["--out_file"])

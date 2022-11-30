# Makefile
# Eric Tsai, Nov 30, 2022

# Script that downloads all necessary files, processes them, and creates a
# regression model to predict the average rating of a boardgame

# example usage:
# make all

.PHONY : all
all : 
	python src/eda_boardgame.py --in_file="data/processed/training_split.csv" --out_dir="results/" 
	python src/prediction_model.py --training_file="data/processed/training_split.csv" --testing_file="data/processed/testing_split.csv" --results_dir="results/"
	doc/boardgame_rating_predictor_report.html


# download the raw ratings file
data/raw/ratings.csv : src/download_data.py
	python src/download_data.py --url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv" --out_file="data/raw/ratings.csv"

# download the raw details file
data/raw/details.csv: src/download_data.py
	python src/download_data.py --url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv" --out_file="data/raw/details.csv"

# preprocess the raw files and combine them into one cleaned file
data/processed : data/raw/ratings.csv data/raw/details.csv src/preprocess_boardgame_data.py
	python src/preprocess_boardgame_data.py --in_file1="data/raw/ratings.csv" --in_file2="data/raw/details.csv" --out_dir="data/processed"

# create the figures from EDA
.PHONY : eda
eda :
	python src/eda_boardgame.py --in_file="data/processed/training_split.csv" --out_dir="results/"

# create the regression model by fitting on different models and selecting the best one
.PHONY : predict_model
predict_model :
	python src/prediction_model.py --training_file="data/processed/training_split.csv" --testing_file="data/processed/testing_split.csv" --results_dir="results/"

# render the final report
doc/boardgame_rating_predictor_report.html : doc/boardgame_refs.bib doc/boardgame_rating_predictor_report.Rmd
	Rscript -e "rmarkdown::render('doc/boardgame_rating_predictor_report.Rmd')"

# delete all generated files and figures
.PHONY : clean
clean: 
	rm -rf data
	rm -f results
	rm -f doc/boardgame*
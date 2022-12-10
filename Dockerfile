# Author Ashwin Babu 12/8/22
FROM jupyter/scipy-notebook

USER root

RUN pip install docopt-ng==0.8.1 \
    && pip install vl-convert-python==0.5.0
RUN conda install -c conda-forge -y pandoc
RUN pip install joblib --quiet

RUN conda update -n base -c conda-forge -y conda

RUN apt-get update
RUN apt-get -y --no-install-recommends install
RUN apt-get install r-base r-base-dev -y
RUN conda install -c conda-forge r r-essentials

RUN conda install python-graphviz -y \
    && conda install requests[version='>=2.24.0'] -y \
    && conda install scikit-learn -y \
    && conda install selenium[version='<4.3.0'] -y \
    && conda install pip -y \
    && conda install jinja2 -y \
    && conda install ipykernel -y \
    && conda install jsonschema=4.16 -y \
    && conda install -c conda-forge altair_saver -y \
    && conda install pandas[version='<1.5'] -y \
    && conda install matplotlib[version='>=3.2.2'] -y \
    && conda install -c conda-forge eli5 -y \
    && conda install -c conda-forge shap -y \
    && conda install -c conda-forge imbalanced-learn -y 

RUN Rscript -e "install.packages('tidyverse', repos='https://cran.rstudio.com/')"
RUN Rscript -e "install.packages('broom', repos='https://cran.rstudio.com/')"

RUN conda install -c conda-forge --quiet --yes \
    'r-rmarkdown' \
    'r-knitr' \
    'r-kableextra'
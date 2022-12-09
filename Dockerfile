# Author Ashwin Babu 12/8/22
FROM jupyter/scipy-notebook

RUN conda install -c conda-forge --quiet --yes \
    'docopt==0.6.*' 

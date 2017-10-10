# Twitter ADR Classification
COMP90049 Knowledge Technologies, Semester 2 2017 Project 2 Assignment - ADR classification for twitter text

## Overview
This project use Jupyter notebook to implement several machine learning classification technique to classify twitter text for ADR context.

Main process in the jupyter notebook
* Implementing several machine learning classifier for initial evaluation against dev data set.
  * Gaussian Naive Bayes
  * Multinomial Naive Bayes
  * Decision Tree
  * Random Forest
  * Support Vector Machine
* Feature Engineering
  * ADR Lexicon
  * Negative Sentiment
* Evaluation against dev data set with new generated features
* Test Data Prediction
* Generating new arff file with new features

### Structure
    ├── data              # all of data for analysis
    ├──── dev             # development set
    ├──── new_features    # all data set with new generated features
    ├──── test            # test data set
    ├──── train           # training set
    ├──── ADR_lexicon.tsv # Lexicon file for generating ADR lexicon features
    ├──── README.txt      # README for data
    ├── README.md         
    └── requirements.txt  # python package dependencies

The prediction results for testing data is in `data/new_features/test.arff` along with `dev` and `training` data with new generated features.

## Getting Started

Make sure you have python2.7 installed in your system.

### Setup Python and VirtualEnv
VirtualEnv is a way to create isolated Python environments for every project and VirtualEnvWrapper "wraps" the virtualenv API to make it more user friendly.

```bash
$ pip install pip --upgrade
$ pip install virtualenv
$ pip install virtualenvwrapper
```

To complete the virtualenv setup process, put the following in your ~/.bash_profile
```bash
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

### Create VirtualEnv and Install Dependencies
The following commands will ensure you have the Python dependencies installed inside your `virtualenv`.

```bash
mkvirtualenv twitter-adr --python=python2
pip install -r requirements.txt
```

## Running The Program

The author use jupyter notebook to do the analysis. We need to firstly run the notebook, and open `Analysis.ipynb` file from jupyter notebook

```
$ workon lexical-twitter
$ jupyter notebook
```
Run each line in the notebook to see the analysis process
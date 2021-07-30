# Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


---

# Overview <a name="overview"></a>

In this project we analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages.


# Installation <a name="installation"></a>

This project uses Python 3.8.3. The following libraries are necessary for running the files: 

- numpy == 1.21.1
- pandas == 1.3.1
- re == 2.2.1
- pickle == 4.0
- nltk == 3.6.2
- scikit-learn == 0.24.2
- SQLAlchemy == 1.4.22
- flask == 0.12.5
- plotly == 2.0.15

# Project Motivation <a name="motivation"></a>

The main motivation was to apply all the knowledge from the Udacity DSNP. It was a challenge at first, but after trying and making mistakes, the final goal was achived.

# File Descriptions <a name="files"></a>

The following is the distribution of the most important files

```
disaster_response_pipeline
    |-- app
          |-- templates
                  |-- go.html
                  |-- master.html
          |-- run.py
    |-- data
          |-- disaster_message.csv
          |-- disaster_categories.csv
          |-- disaster_process_data.db
          |-- process_data.py
    |-- models
          |-- train_classifier.py
          |-- classifier.pkl
    |-- notebooks
          |-- 1. ETL Pipeline Preparation.ipynb
          |-- 2. ML Pipeline Preparation.ipynb

```
There is also the `plot` folder that contains different graphics of the platform interface and the results. On the other hand, the `project_env` folder is a virtual environment created specially for this project.


# Results <a name="results"></a>

The most important result is the web app that was created. In the "plot" folder you can see the interface and how it works.

Getting started: 

1) Clone this repository
```
git clone https://github.com/alexrop/disaster_response_figure_eight.git
```

2) Run the program

```
python run.py
```

3) Web App interface

On terminal type: 
```
env|grep WORK
```

Then go to the following address by completing with your own *SPACEDOMAIN* and *SPACEID*

```
https://SPACEID-3001.SPACEDOMAIN
```
Now, if you want to re-execute the `process_data.py` and `train_classifier.py` scripts, the nomenclature are:

```
python process_data.py disaster_messages.csv disaster_categories.csv disaster_process_data.db
```

```
python train_classifier.py ../data/disaster_process_data.db classifier.pkl
```
The last one takes about 2 h aprox to be completed


On the other hand, if you need more details about the ETL and ML Pipeline process, check the *notebook* folder,  where you can find a "step-by-step" explination about how we generate the final database and model.


# Licensing, Authors, Acknowledgements <a name="licensing"></a>

All data here belongs to the [Figure Eight](https://appen.com/) company, and it was provided by the [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025) team.
Author: Alexander Ulloa Opazo

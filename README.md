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

This project uses Python 3.6.3. The following libraries are necessary for running the files: 

- numpy==1.12.1
- pandas==0.23.3
- re==2.2.1
- pickle==4.0
- nltk==3.2.5
- scikit-learn==0.19.1
- SQLAlchemy==1.2.19
- flask==0.12.5
- plotly==2.0.15

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

Then go to the following address by completing with your own *SPACEDOMAIN* and *SPACEID*. For example, in my case was *https://view6914b2f4-3001.udacity-student-workspaces.com/*

```
https://SPACEID-3001.SPACEDOMAIN
```
Now, if you want to re-execute the `process_data.py` and `train_classifier.py` scripts, you must follow this nomenclature:

```
python process_data.py disaster_messages.csv disaster_categories.csv disaster_process_data.db
```

```
python train_classifier.py ../data/disaster_process_data.db classifier.pkl
```
>The last one (train_classifier.py) takes around 2 h aprox to be completed


On the other hand, if you need more details about the ETL and ML Pipeline process, check the *notebook* folder,  where you can find a "step-by-step" explination about how we generate the final database and model.

> Note: This project was made on the *Udacity Project Workspace IDE platform* (Ubuntu based) so it might be some differences if you run it on Windows. Check that the package versions are the same that mine as I showed you above.

![image](https://user-images.githubusercontent.com/49656060/127631805-febda984-8554-4906-9c81-efd6525f2c04.png)


# Licensing, Authors, Acknowledgements <a name="licensing"></a>

All data here belongs to the [Figure Eight](https://appen.com/) company, and it was provided by the [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025) team.
Author: Alexander Ulloa Opazo

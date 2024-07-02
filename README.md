# Disaster Response Classifier
This repository contains a classifier for disaster response messages.

## Table of contents
- [Project Motivation](#project-motivation)
- [File Descriptions](#file-descriptions)
- [Get started](#get-started)
- [Creator](#creator)
- [Thanks](#thanks)

## Project Motivation
As part of the Udacity Nanodegree for Data Science, I developed a classifier based on pre-labeled data that was
provided by [Appen](https://www.appen.com). The model is accessible via a web interface to instantly classify
messages.

## File Descriptions
- `/app/run.py` - Python script that starts and controls the Flask Web App
- `/app/templates/master.html` - Main page that contains statistics of the training data and a formular for 
messages that will be classified
- `/app/templates/classify_message.html` - Page with classification results for the requested message
- `/data/disaster_messages.csv` - Over 26k messages that were pre-labeled by 36 disaster categories
- `/data/disaster_categories.csv` - Pre-labeled disaster categories
- `/data/process_data.py` - Python script that merge, clean and transform the disaster csv data
- `/data/etl.db` - Output data of the process_data.py stored as a SQLite database
- `/model/train_classifier.py` - Python script to train the model that classifies messages by disaster category
- `/model/model.pkl` - Trained model stored as Pickle file (not uploaded to GitHub)
- `/requirements.txt` - List of used python libraries

## Get started
- Install [python](https://www.python.org/downloads/) (used version: 3.9.13)
- The required python libraries are listed in the [requirements.txt](https://github.com/PatrickChristoph/disaster_response/tree/main/requirements.txt)
- ETL Pipeline: To process the ETL pipeline you have to run the [process_data.py](https://github.com/PatrickChristoph/disaster_response/tree/main/data/process_data.py)
- ML Pipeline: To train the model based on the created database of the ETL pipeline you have to run the [train_classifier.py](https://github.com/PatrickChristoph/disaster_response/tree/main/data/train_classifier.py)
- Web App: Run the web app locally by executing the [run.py](https://github.com/PatrickChristoph/disaster_response/tree/main/app/run.py) and follow the instructions

## Creator

**Patrick Christoph**
- <https://www.linkedin.com/in/patrick-christoph/>
- <https://github.com/PatrickChristoph>

## Thanks

Local Test Branch - remove

Meanwhile, main changes something...

and something more

<a href="https://www.udacity.com/">
  <img src="https://www.udacity.com/images/svgs/udacity-tt-logo.svg" alt="Udacity" width="192" height="48">
</a>

Thanks to [Udacity](https://www.udacity.com/) for the introduction course in data science.

<a href="https://www.appen.com/">
  <img src="https://companieslogo.com/img/orig/APX.AX_BIG-d8c7efa0.png" alt="Appen" height="48">
</a>

Thanks to [Appen](https://www.appen.com/) for providing the data.

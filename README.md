# QRT Football Prediction Challenge

## Description

The aim of this project is to predict the results of football matches, given historical data of the two teams at the season and last 5 matches level. 

## Overview
The main problem to contend with during this project was the high degree of noise in the training set. As such data cleaning and feature engineering played an important part in the approach I took. The following steps were taken:

- **Data Exploration**: Looked for correlation between different features. The mean data was highly correlated with the sum and so we removed all of these columns. 

- **Clean data by imputing all missing values**. Where possible this was done by taking estimates from other parts of the team data set, e.g. estimating season injuries with last 5 match injuries (accounting for different scale). If this was not possible, I imputed the mean value.

- **Feature Engineering**:
	- *Difference Features*: As I wanted to compare teams, I generated 'diff' features, which measured the home sum minus the away sum. I did this for both the season level data and the last 5 match data. 
	- *Prior Result*: This was a simple estimate that the game chosen was a win, draw or loss for the home team, given the number of games both the home and away team had WLD. 
	- *Good Shot Metric*: The shot metrics were highly correlated and I used linear regression to calculate the correct waiting. To do this, I linearly regressed the metrics onto the relevant number of goals. 
	- *Form Metric*: Took all the data for the last 5 matches and using to predict a variable composed of $2\times\text{number of wins in last 5} +1\times\text{number of draws in last 5} + 0\times\text{number of loss in last 5}$ using linear regression. This gave a measure of 'form'.
	- *Predicted Goal Diff*: Estimated the goal difference using the supplementary training set. To do this I selected the most important parameters (using stepwise forward selection) and training a variety of regression models on the data. 

- **Result Prediction**: Used forward stepwise selection to select the most relevant features. Then trained and tuned a variety of different classifiers to predict if there was a win or loss. We found that when training, WLD performed worse on the training set and so we limited ourselves to WL models only. 

## Results 

Once the training was done, the correct results were predicted 48% of the time. On validation data sets, I was achieving 51% accuracy and so I believe the result could be improved by further removal of features and limiting overfitting.    


## Notebook Overview
The following notebooks perform the following tasks:

- `Exploratory_Data_Analysis`: Does the initial data exploration and cleaning. Uses models created in other notebooks to generate new features. Saves the training set to file. 
- `Form_Feature`: Does the linear regression of all the last 5 matches statistics to find a good weighting between the different features.
- `Generate_Predictions`: Reads in the fitted WL model and makes predictions. Saves these predictions in the right format
- `Goal_Diff_Prediction`: Selects the most important features and trains a variety of different regression models to predict the goal difference. 
- `Good_Shot_Feature`: Linearly regresses all the 'shot' metrics onto the relevant 'goal' metric in order to find a good average weighting in the number of shots. 
`WL_Result_Prediction`: Selects the most relevant features for predicting the result of a game. The trains binary classification model to predict the result. 
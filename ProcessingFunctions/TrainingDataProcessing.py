from ProcessingFunctions.ImputingMissingValues import fill_missing_values, contains_missing_values
from ProcessingFunctions.FeatureEngineering import generate_features
from ProcessingFunctions.RemoveFeatures import remove_extra_features

import pandas as pd 
import numpy as np

def creating_joint_df(home_training_file : str, away_training_file : str, remove_team_names : bool):
	'''This function takes the home team statistics and the away team statistics and combines them '''
	train_home = pd.read_csv(home_training_file, index_col=0)
	train_away = pd.read_csv(away_training_file, index_col=0)	

	if remove_team_names:
		train_home = train_home.iloc[:,2:]
		train_away = train_away.iloc[:,2:]

	train_home.columns = 'HOME_' + train_home.columns # We do this so that we can combine the data sets into one for training
	train_away.columns = 'AWAY_' + train_away.columns

	train_data = pd.concat([train_home, train_away], join = 'inner', axis = 1)
	train_data = train_data.replace({np.inf:np.nan,-np.inf:np.nan})

	return train_data


def create_processed_training_df(home_training_file, away_training_file, remove_team_names):
	''' This pulls together all the different processing steps '''
	df = creating_joint_df(home_training_file, away_training_file, remove_team_names)
	df = fill_missing_values(df)
	df = generate_features(df)
	df = remove_extra_features(df)

	return df 


if __name__ == '__main__': 
	df = create_processed_training_df('Train_Data/train_home_team_statistics_df.csv','Train_Data/train_away_team_statistics_df.csv',True)		


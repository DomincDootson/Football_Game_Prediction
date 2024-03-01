'''This file contains all the functions in order to generate new features, namely:
	1. Difference Features (Run this before GD prediction, but aftwer PF, SGF, FF)
	2. Prior feature
	3. Good shot feature
	4. Form feature
	5. Goal difference predictions
'''
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

## Generate Features ##
## ----------------- ##

def generate_features(df):
	''' Generate all the features, could us a pipeline'''
	df = generate_difference_features(df)
	df = generate_prior_features(df)
	df = generate_good_shot_features(df)
	
	df = generate_form_feature(df)
	df = generate_goal_difference_feature(scale_values(df))

	return scale_values(df)


## Difference Features ##
## ------------------- ##

def generate_difference_features(df):
	''' Generates the difference column'''
	home_col, away_col = get_home_away_cols(df.columns)
	diff_col = ["DIFF_" + "_".join(c.split('_')[1:]) for c in home_col if ('std' not in c)]
	
	df_home, df_away = df[home_col].copy(), df[away_col].copy()
	
	df_home = df.rename(columns = {hc : dc for hc, dc in zip(home_col, diff_col)})
	df_away = df.rename(columns = {ac : dc for ac, dc in zip(away_col, diff_col)})

	df[diff_col] = (df_home[diff_col] - df_away[diff_col]).copy()
	
	return scale_values(df)


## Prior Features ##
## -------------- ##

def generate_prior_features(df):
	'''Generates the prior features and adds them to the df'''
	home_matches = df[['HOME_TEAM_GAME_WON_season_sum','HOME_TEAM_GAME_DRAW_season_sum','HOME_TEAM_GAME_LOST_season_sum']].sum(axis=1)
	away_matches = df[['AWAY_TEAM_GAME_WON_season_sum','AWAY_TEAM_GAME_DRAW_season_sum','AWAY_TEAM_GAME_LOST_season_sum']].sum(axis=1)

	HW = df['HOME_TEAM_GAME_WON_season_sum']/home_matches
	HD = df['HOME_TEAM_GAME_DRAW_season_sum']/home_matches
	HL = df['HOME_TEAM_GAME_LOST_season_sum']/home_matches

	AW = df['AWAY_TEAM_GAME_WON_season_sum']/away_matches
	AD = df['AWAY_TEAM_GAME_DRAW_season_sum']/away_matches
	AL = df['AWAY_TEAM_GAME_LOST_season_sum']/away_matches

	df['WIN_PRIOR'] = HW * AL
	df['DRAW_PRIOR'] = HD * AD
	df['LOST_PRIOR'] = HL * AW

	return df

## Good Shot Features ##
## ------------------ ##

def get_good_shot_col(cols : list[str], HorA : str, Sor5 : str) -> list[str]:
	'''Returns different shot columns '''
	return [c for c in cols if (HorA in c and Sor5 in c and 'SHOT' in c and 'sum' in c)]

def generate_good_shot_features(df, model_file = 'Good_Shot_Feature/good_shot_model.pkl'):
	''' Adds the good shot features to the model, note it doesnt add the diff good shot'''
	good_shot_model = read_in_model(model_file)
	
	home_shot_season = get_good_shot_col(df.columns, 'HOME', 'season')
	home_shot_5 = get_good_shot_col(df.columns, 'HOME', '5_last_match')
	away_shot_season = get_good_shot_col(df.columns, 'AWAY', 'season')
	away_shot_5 = get_good_shot_col(df.columns, 'AWAY', '5_last_match')

	df['HOME_GOOD_SHOT_season_sum'] = good_shot_model.predict(df[home_shot_season])
	df['HOME_GOOD_SHOT_5_last_match_sum'] = good_shot_model.predict(df[home_shot_5])
	df['AWAY_GOOD_SHOT_season_sum'] = good_shot_model.predict(df[away_shot_season])
	df['AWAY_GOOD_SHOT_5_last_match_sum'] = good_shot_model.predict(df[away_shot_5])

	df['DIFF_GOOD_SHOT_season_sum'] = df['HOME_GOOD_SHOT_season_sum'] - df['AWAY_GOOD_SHOT_season_sum']
	df['DIFF_GOOD_SHOT_5_last_match_sum'] = df['HOME_GOOD_SHOT_5_last_match_sum'] - df['AWAY_GOOD_SHOT_5_last_match_sum']

	return df

## Form Metric ##
## ----------- ##

def generate_form_feature(df, model_file = 'Form_Feature/form_model.pkl', features_file = 'Form_Feature/form_features.csv'):
	''' Generate the form feature '''
	form_model = read_in_model(model_file)
	form_features = read_in_features(features_file)

	away_form_features, home_form_features = ['AWAY_'+f for f in form_features], ['HOME_'+f for f in form_features]

	df['AWAY_FORM_5_last_match_sum'] = form_model.predict(df[away_form_features].rename(columns = {o : n for o, n in zip(away_form_features, form_features)}))
	df['HOME_FORM_5_last_match_sum'] = form_model.predict(df[home_form_features].rename(columns = {o : n for o, n in zip(home_form_features, form_features)}))
	df['DIFF_FORM_5_last_match_sum'] = df['HOME_FORM_5_last_match_sum'] - df['AWAY_FORM_5_last_match_sum']
	return df.fillna(df.mean())

## GD Feature ## 
## ---------- ##

def generate_goal_difference_feature(df, model_file = 'Goal_Diff_Prediction/normal_sampled_DG_predictor.pkl', features_file = 'Goal_Diff_Prediction/Features.csv'):
	gd_model = read_in_model(model_file)
	gd_features = read_in_features(features_file)
	
	df['NORMAL_SAMPLED_PREDICTED_GD'] = gd_model.predict(df[gd_features])

	return df


## Misc ##
## ---- ##

def get_home_away_cols(cols):
	return [c for c in cols if 'HOME' in c], [c for c in cols if 'AWAY' in c]

def read_in_model(model_file):
	with open(model_file, 'rb') as file:
		model = pickle.load(file) 

	return model 

def read_in_features(features_file) -> list[str]:
	return pd.read_csv(features_file)['Feature'].to_list()

def scale_values(df):
	scaler = MinMaxScaler(feature_range = (0, 10))
	return pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
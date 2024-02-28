import pandas as pd
from operator import lt, gt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
class ForwardsStepwiseSelection():
	"""docstring for ForwardsStepwiseSelection"""
	def __init__(self, p, model, metric, increasing_is_better : bool = False):
		
		self.p = p
		self.model = model
		self.metric = metric
		
		self.starting_value = -float('inf') if increasing_is_better else float('inf')
		
		self.is_better_score = gt if increasing_is_better else lt

		self.training_score, self.valid_score = None, None

		

	def fit(self, X, y, validation_set = None, verbose = True):
		features, self.best_features = set(X.columns), []
		self.training_score, self.valid_score = [], []
		
		for i in range(self.p):
			if verbose:
				print(f"Finding the {i}th feature")
			best_new_feature = ''
			best_valid_score, best_train_score = self.starting_value, 0
			for new_feature in (f for f in features if f not in self.best_features):
				features_2_try = self.best_features + [new_feature]
				print(features_2_try)
				print()
				
				self.model.fit(X[features_2_try], y) # Check that we use different features each time 
				
				test_score  = self.metric(y, self.model.predict(X[features_2_try]))
				valid_score = self.metric(validation_set[1], self.model.predict(validation_set[0][features_2_try]))
				
				if self.is_better_score(valid_score, best_valid_score):
					best_new_feature = new_feature
					best_valid_score = valid_score
					best_train_score = test_score


			self.best_features.append(best_new_feature)
			self.training_score.append(best_train_score)
			self.valid_score.append(best_valid_score)

	def rank_features(self) -> None:
		print(*enumerate(self.best_features), sep = '\n')

	def plot_error(self):
		plt.plot(self.best_features, self.training_score, label = 'Training Score')
		plt.plot(self.best_features, self.valid_score, label = 'Validation Score')

		plt.xticks(rotation=90)
		plt.legend()
		plt.show()



	## Some method to plot the errors 
if __name__ =='__main__':

	X = pd.read_csv('../goal_diff_prediction/GD_training_data.csv')
	y = pd.read_csv('../Data/Y_train_supp.csv', index_col = "ID")

	fss = ForwardsStepwiseSelection(20, LinearRegression(), mean_squared_error, False)
	X_train, X_valid, y_train, y_valid = train_test_split(X,y, random_state = 48,test_size=0.2)

	fss.fit(X =X_train,y=y_train, validation_set = [X_valid, y_valid])
	fss.rank_features()

	fss.plot_error()
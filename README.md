# QRT Football Predicition Challenge



## Notebook Overviews

### Exploratory_Data_Analysis

The aim of this notebook is to contain all data clean, exploration of relevant features and ultimately to pull 
together our novel features. In particular, we complete the following steps:
- We remove all missing values. First, we use similar features to estimate missing values, e.g. taking last 5 
match injury sum as an estimate for the season injury sum. Any values still missing we impute with the mean value.
- Plot the correlation between different features. We find that the mean and sum metrics are highly 
correlated, leading us to (for features that have both sum and mean values), remove the features representing 
the mean values. We look at correlation between the season values and last 5 matches and find it to be high (motivating the first imputation 
step that we take).
- Feature engineering. There are many features, and a lot of noise. We generate new features so that when 
fitting the data, we can use fewer features to capture the same infomation, reducing the worry of 
overfitting. In particular, we generate:
	- The most drastic step we take is to generate `DIFF` feature for each of the sum values which is 
calculated as `HOME_FEATURE_..._sum`-`AWAY_FEATRUE_..._sum`. We are ultimately interested in the 
comparison between home and away and this is the easiest way to include it. Some features we leave in as 
'absolute' values, e.g. HOME_WIN as this is an important number both relative to AWAY_WIN and absolutely. 
	- We generate a 'prior' of winning/drawing/loosing. This can be consider as the probablity that if we were 
to select a random game from the home and away's team season that they are a W&L, D&D or L&W. This is not 
mathematically rigorious.
	- We fit the total goals features with all a linear model, using all the shot measures as features. This 
allows us to combine the shot features, with sensible propotions (i.e. it answers the question, 'how much more 
important are on target shots vs off target etc.).
	- We create a target of 2*(last 5 wins) 1*(last 5 draws) +0*(last_5 loss) as a measure of form, and fit it 
with all the last 5 features, as a way to amlgamate the different features. 
	- Using all these features, we fit a regressive model to predict the goal difference (given in the 
suplimentary data).
- We rescale all the data, and then save to file.  

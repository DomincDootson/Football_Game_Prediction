'''This file contains all the functions that I use to remove features'''
KEEP_HOME_AWAY_VALUES = { 'TEAM_GAME_DRAW_season', # Keep in absolute value
                'TEAM_GAME_LOST_season', # Keep in absolute value
                'TEAM_GAME_WON_season', # Keep in absolute value
                'TEAM_INJURIES_season'}


def should_keep_col_avg(s):
    if 'average' not in s:
        return True
    if ('TEAM_BALL_POSSESSION' in s) or ('TEAM_SUCCESSFUL_PASSES_PERCENTAGE' in s):
        return True
    return False

def remove_averages(df):
    cols_without_avg = [c for c in df.columns if should_keep_col_avg(c)]   
    return df[cols_without_avg].copy()



def should_keep_remove_sum(s, others_to_keep):
    split_s = s.split('_')
    for term in others_to_keep:
        if term in s:
            return True
    
    if split_s[0] != 'DIFF' and (split_s[-1] == 'sum' or split_s[-1] == 'average'):
        return False
    return True
    

def remove_HOME_AWAY_sums(df):
    cols_without_avg_sum = [c for c in  df.columns if should_keep_remove_sum(c, KEEP_HOME_AWAY_VALUES)]
    return df[cols_without_avg_sum].copy()


def remove_extra_features(df):
    return remove_HOME_AWAY_sums(remove_averages(df))



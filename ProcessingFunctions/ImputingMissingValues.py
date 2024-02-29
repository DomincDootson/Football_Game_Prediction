''' This file contains the functions for dealing with missing values '''
import pandas as pd
import numpy as np 

def fill_values_with_swaps(df, col1, col2):
    ''' Given a data frame, this coloumn fills in any missing values 
    with those taken from a different column and vice versa '''
    
    df[col1] = df[col1].fillna(df[col2])
    df[col2] = df[col2].fillna(df[col1])

    return df

def is_pairing_correct(pair : tuple[str,str]) -> bool:
    '''Checks if we correctly paried the season and last 5 matches col'''
    return ' '.join(pair[0].split('_')[:-2]) == ' '.join(pair[0].split('_')[:-4])

def get_season_and_last_5_cols(cols : list[str]) -> list[tuple[str, str]]:
    '''This will return return a list where the season and last 5 matches col titles are paired.
    It assumes that the cols are in the order home season, home last 5, away season, away last 5 '''

    season_cols = [c for c in cols if 'season' in c]
    last_5_cols = [c for c in cols if '5_last_match' in c]

    paired_cols = [(season, last_5) for season, last_5 in zip(season_cols, last_5_cols)]

    for p in paired_cols:
        if is_pairing_correct(p):
            print("Pairing hasn't worked")
            exit()

    return paired_cols


def swap_last_5_and_col_missing_values(df):
    '''Uses last 5 to estimate season missing values and vice versa'''
    paired_cols = get_season_and_last_5_cols(df.columns) 
    for s, l in paired_cols:
        df = fill_values_with_swaps(df, s, l)

    return df 

def fill_final_missing_with_mean(df):
    '''read the function name...'''
    return df.fillna(df.mean())

def fill_missing_values(df):
    ''' Does the swaping and mean imputation of the data'''
    df = swap_last_5_and_col_missing_values(df)
    return fill_final_missing_with_mean(df)

def contains_missing_values(df):
    ''' Returns true if the array has any missing values'''
    return df.isnull().any().any()
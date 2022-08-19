import numpy as np 
import pandas as pd
# REMOVED MODULE IMPORT STATEMENTS


# Modified example work script of a machine learning Python data pipeline.
# This script is intended as a work example of Adam Morphy's work with the Vancouver Whitecaps FC, and has been modified outside its original data pipeline.
# Data names and references have been removed.

#################################################
#                   _______
#                  |       |              
#            o     |       |
#          -()-   o
#           |\
#################################################


def super_read_df(match_list):
    # call in two df to be used
    df = get_supervised(match_list)
    df = df.sort_values(by=['COL1'])
    df_touch = get_touch_data(match_list)
    return df, df_touch


def join_touch(df, df_touch):
    # join df and df_touch using touch_id
    df = pd.merge(df, df_touch, how='left', on='COL1')
    return df


def super_build_df(df):
    """
    Cleans and prepares the DataFrame for supervised machine learning models

    Parameters
    ----------
    df : DataFrame
        Raw DataFrame before conversion

    Returns
    ----------
    df: DataFrame 
        Finalized DataFrame after conversion
    """

    # create a new column for pass type
    conditions = [
        (df['COL1'] == 1) & (df['COL2'] == 0) & (df['COL3'] == 0) & (df['COL4'] == 0),
        (df['COL5'] == 1),
        (df['COL6'] == 1),
        (df['COL7'] == 1)
    ]
    choices = ['COL1', 'COL2', 'COL3', 'COL4']
    df['COL1'] = np.select(conditions, choices, default='COL1')

    # calculate the pass distance based x,y coordinates of origin and target
    df['COL9'] = np.sqrt(((df['COL1'].values - df['COL2'].values)**2) + ((df['COL3'].values - df['COL4'].values)**2))

    # create primary key with match & frame id
    df['COL5'] = df['COL6'].astype(str) + df['COL7'].astype(str)

    # drop unused columns
    df = df.drop([
        # COLS
    ], axis=1)

    # drop duplicate rows by match_frame_id
    df = df.drop_duplicates(subset='COL2')

    # reset index
    df = df.reset_index(drop=True)

    return df

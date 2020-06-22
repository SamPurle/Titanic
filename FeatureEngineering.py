"""

Titanic - Feature Engineering:
    
    A script to extract additional information from the available data.
    
"""

# Import libraries

import pandas as pd


"""
Engineer: Initial, basic feature Engineering to extract additional meaninng from other columns.
"""


def Engineer(df):
    
    df['FamSize'] = df['Parch'] + df['SibSp']
    
    df['Alone'] = 0
    df.loc[df['FamSize'] == 0, 'Alone'] = 1
    
    return df


"""
Binning: Binning existinng data to reduce noise within the dataset
"""


def Bin(df):
    
    df['AgeBin'] = pd.cut(df['Age'], bins = [0, 5, 13, 20, 45, 65, 100], 
                          labels = False)
    df.drop(columns = 'Age', inplace = True)
       
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels = False) 
    df.drop(columns = 'Fare', inplace = True)
    
    return df
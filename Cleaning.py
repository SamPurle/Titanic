"""

Titanic - Cleaning:
    
    A script to clean the Raw data intto a usable format
    
"""

# Import libraries

import pandas as pd
from sklearn.impute import SimpleImputer

# Load data

dfTrain = pd.read_csv('D:/Datasets/Titanic/train.csv', index_col ='PassengerId')
global TrainIndex
TrainIndex = dfTrain.index

dfTest = pd.read_csv('D:/Datasets/Titanic/test.csv', index_col = 'PassengerId')
global TestIndex
TestIndex = dfTest.index

# Concatenate to allow simultaneous cleaning

df = pd.concat([dfTrain, dfTest])

# Identify null percentages

NullPer = df.isnull().sum() / len(df)
NullPer.sort_values(ascending = False, inplace = True)

# Drop useless x columns

global NULL_THRESH
NULL_THRESH = 0.25  # Specify the threshold for dropping null columns

NullCols = NullPer.loc[NullPer > NULL_THRESH].index
NullCols = NullCols.drop('Survived') # Do not drop the y column from the dataset

df.drop(columns = NullCols, inplace = True)

# Drop columns containing meaningless information

df.drop(columns = 'Ticket', inplace = True)

# Identify remaining Nulls

NullPer = df.isnull().sum() / len(df)
NullPer.sort_values(ascending = False, inplace = True)
NullPer = NullPer.drop('Survived')

NullCols = NullPer.loc[NullPer > 0].index

# Identify categorical and numeric null columns

CatCols = df.columns.drop(df._get_numeric_data().columns)

NumCols = df._get_numeric_data().columns

# Impute null values

ImpMode = SimpleImputer(strategy = 'most_frequent', copy = False)
ToImputeMode = CatCols.join(NullCols, how = 'inner')

for c in ToImputeMode:
    df[c] = ImpMode.fit_transform(df[[c]])
    
ImpMean = SimpleImputer(strategy = 'mean')
ToImputeMean = NumCols.join(NullCols, how = 'inner')

for c in ToImputeMean:
    df[c] = ImpMean.fit_transform(df[[c]])
    
# Save data

df.to_csv('D:/Datasets/Titanic/CleanedData.csv')


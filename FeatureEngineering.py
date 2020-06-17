"""

Titanic - Feature Engineering:
    
    A script to extract additional information from the available data.
    
"""

# Import libraries

import pandas as pd

# Load data

df = pd.read_csv('D:/Datasets/Titanic/CleanedData.csv', index_col = 'PassengerId')

# Add new columns

df['FamSize'] = df['Parch'] + df['SibSp']

df['Alone'] = 0
df.loc[df['FamSize'] == 0, 'Alone'] = 1

# Save data

df.to_csv('D:/Datasets/Titanic/CleanedData.csv')

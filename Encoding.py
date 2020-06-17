"""

Titanic - Encoding:
    
    A script to handle categorical columns within the dataset
    
"""

# Import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Load data

df = pd.read_csv('D:/Datasets/Titanic/CleanedData.csv', index_col = 'PassengerId')

# Extract title from name

df['Title'] = df['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]
df.drop(columns = 'Name', inplace = True)

# Group rare titles into one category

TitleCounts = df.groupby(['Sex', 'Title']).Title.count()
print(TitleCounts)

FancyTitles = TitleCounts.loc[TitleCounts.values < 10]

MaleFancy = FancyTitles.loc['male']
MaleIndexReplace = df.join(MaleFancy, on = 'Title', how = 'inner', rsuffix = 'F').index 
df.loc[MaleIndexReplace, 'Title'] = 'FancyMan'

FemaleFancy = FancyTitles.loc['female']
FemaleIndexReplace = df.join(FemaleFancy, on = 'Title', how = 'inner', rsuffix = 'F').index 
df.loc[FemaleIndexReplace, 'Title'] = 'FancyLady'

# Find categorical columns

CatCols = df.columns.drop(df._get_numeric_data().columns)

# Plot to determine whether one-hot-encoding is necessary

for x in CatCols:
    sns.barplot(x = x, y = 'Survived', data = df)
    
    plt.title('Plot showing percentage survival based on {}'.format(x))    
    plt.show()
    
# Encoding

Enc = OneHotEncoder(sparse = False)

for x in CatCols:
    EncCol = Enc.fit_transform(df[[x]])
    EncNames = Enc.get_feature_names([x])
    EncDf = pd.DataFrame(EncCol)
    EncDf.columns = EncNames
    
    df = df.join(EncDf)    
    df.drop(columns = x, inplace = True)

df.to_csv('D:/Datasets/Titanic/CleanedData.csv')
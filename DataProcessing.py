"""

Titanic - Data Processing:
    
    A collection of functions to convert the raw data into a more usable
    format.
    
"""

# Import libraries

from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


"""
Cleaning: A script to clean the Raw data into a usable format
"""    


def Clean(df, NULL_THRESH):
    
    # Identify null percentages
    
    NullPer = df.isnull().sum() / len(df)
    NullPer.sort_values(ascending = False, inplace = True)
        
    # Drop useless x columns
    
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
        
    return df


"""
Encoding: A script to handle categorical columns within the dataset
"""


def Encode(df, FANCY_THRESH):
    
    # Extract title from name

    df['Title'] = df['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]
    df.drop(columns = 'Name', inplace = True)
    
    # Group rare titles into one category
    
    TitleCounts = df.groupby(['Sex', 'Title']).Title.count()
        
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
        EncDf.index = df.index
        
        df = df.join(EncDf)    
        df.drop(columns = x, inplace = True)
   
    return df


"""
Split: A function to split the cleaned dataframe into train and test sets
"""


def Split(df, TrainIndex, TestIndex):
    dfTrain = df.loc[TrainIndex]
    dfTest = df.loc[TestIndex]
    
    yTrain = dfTrain['Survived']
    dfTrain.drop(columns = 'Survived', inplace = True)
    

    
    return dfTrain, dfTest
    
    
"""
Output: A function to output data into a format suitable for submission
"""


def Output(data, filename, TestIndex):
    dfOutput = pd.DataFrame(data)
    dfOutput.columns = ['Survived']
    dfOutput.index = TestIndex    
    dfOutput.to_csv('Submissions/{}.csv'.format(filename))
    
    
"""
Plot: A function to plot the utilised variables prior to encoding
"""

def Plot(df):
    
    CatCols = df.columns.drop(df._get_numeric_data().columns)
    NumCols = df._get_numeric_data().columns
    
    for x in CatCols:
        sns.barplot(x = x, y = 'Survived', data = df)
        plt.title('Plot showing the impact of {} on passenger survival'.format(x))
        plt.show()

    for x in NumCols:
        sns.regplot(x = x, y = 'Survived', data = df)
        plt.title('Plot showing the impact of {} on passenger survival'.format(x))
        plt.show()

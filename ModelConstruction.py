"""

Titanic - Model Construction:
    
    Construction of an initial machine learning model to predict whether 
    passengers survived the Titanic.
    
"""

# Import libraries

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data

df = pd.read_csv('D:/Datasets/Titanic/CleanedData.csv')

#


SurvModel = RandomForestClassifier()
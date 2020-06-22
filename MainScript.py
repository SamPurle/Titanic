"""

Titanic:
    
    A Machine Learning model to predict whether passengers survived the Titanic.
    
"""

# Import libraries

import time
import pandas as pd
import DataProcessing
import FeatureEngineering
import ModelConstruction

# Load data

dfTrain = pd.read_csv('D:/Datasets/Titanic/train.csv', index_col ='PassengerId')
TrainIndex = dfTrain.index

dfTest = pd.read_csv('D:/Datasets/Titanic/test.csv', index_col = 'PassengerId')
TestIndex = dfTest.index

# Concatenate to allow simultaneous cleaning

df = pd.concat([dfTrain, dfTest])

# Specify parameters

NULL_THRESH = 0.25  # Specify the threshold for dropping null columns

FANCY_THRESH = 10 # Specify the maximum number of people for a title to be considered fancy

TRAIN_SIZE = 0.8 # Specify the proportion of labelled data to be used for training

N_ITER= 100 # Specify the number of parameters to sample during model optimisation

FOLDS = 10 # Specift the number of cross-validation folds

# Run functions

st = time.time()
df = DataProcessing.Clean(df, NULL_THRESH)
print('The completion time for Cleaning was {:.1f} seconds'.format(time.time() - st))

# st = time.time()
# DataProcessing.Plot(df)
# print('The completion ttime for Plotting was {:.1f} seconds'.format(time.time() - st))

st = time.time()
df = DataProcessing.Encode(df, FANCY_THRESH)
print('The completion time for Encoding as {:.1f} seconds'.format(time.time() - st))

st = time.time()
df = FeatureEngineering.Engineer(df)
print('The completion time for Feature Engineering was {:.1f} seconds'.format(time.time() - st))

st = time.time()
yPred = ModelConstruction.Build(df, TrainIndex, TestIndex, TRAIN_SIZE)
print('The completion time for Model Construction was {:.1f} seconds'.format(time.time() - st))

st = time.time()
DataProcessing.Output(yPred, 'InitialModel', TestIndex) # 0.79425
print('The time taken for Data Output was {:.1f} seconds'.format(time.time() - st))

st = time.time()
BestEstimator, yPred = ModelConstruction.Optimise(df, TrainIndex, TestIndex, N_ITER, FOLDS)
print('The completion time for Model Optimisation was {} seconds'.format(time.time() -st))

DataProcessing.Output(yPred, 'OptimisedModel', TestIndex) # 0.79425

df = FeatureEngineering.Bin(df)


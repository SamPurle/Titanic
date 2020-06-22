"""

Titanic - Model Construction:
    
    Construction of an initial machine learning model to predict whether 
    passengers survived the Titanic.
    
"""

# Import libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV


"""
Build: A function to build a basic Random Forest Classifier model
"""


def Build(df, TrainIndex, TestIndex, TRAIN_SIZE):
    
    dfTrain = df.loc[TrainIndex]
    dfTest = df.loc[TestIndex]
    
    yCol = 'Survived'
    
    yTrain = dfTrain[yCol]
        
    xTrain = dfTrain.drop(columns = yCol)
    xTest = dfTest.drop(columns = yCol)
        
    x_Train , x_Test, y_Train, y_Test = train_test_split(
        xTrain, yTrain, random_state = 42, train_size = TRAIN_SIZE)
    
    SurvModel = RandomForestClassifier(oob_score = True, random_state = 42)
    SurvModel.fit(x_Train, y_Train)
    yPred = SurvModel.predict(xTest)
    
    return yPred
    

"""
Optimise: A function to optimise model parameters, using K-fold cross validation accuracy as a performance metric
"""


def Optimise(df, TrainIndex, TestIndex, n_iter, FOLDS):
    
    dfTrain = df.loc[TrainIndex]
    dfTest = df.loc[TestIndex]
    
    yCol = 'Survived'
    
    yTrain = dfTrain[yCol]
    
    xTrain = dfTrain.drop(columns = yCol)
    xTest = dfTest.drop(columns = yCol)
    
    # Specify base model
    
    rfc = RandomForestClassifier(random_state = 42)
    rfc.fit(xTrain, yTrain)
    
    # Specify parameters to vary
    
    EstimatorOptions = range(1,501)
    DepthOptions = range(1,51)
    FeatureOptions = range(1,25)
    ParamGrid = dict(n_estimators = EstimatorOptions, max_depth = DepthOptions, max_features = FeatureOptions)
    
    CVGrid = RandomizedSearchCV(rfc, ParamGrid, n_iter = n_iter, n_jobs = 3,
                       scoring = 'accuracy', cv = FOLDS, random_state = 42, 
                       return_train_score = True)
    CVGrid.fit(xTrain,yTrain)
    
    BestEstimator = CVGrid.best_estimator_
    
    yPred = BestEstimator.predict(xTest)
        
    return BestEstimator, yPred


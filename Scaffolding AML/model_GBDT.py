import lightgbm as lgb
from scipy.stats import randint

def getModel():
    # create GBDT model
    gbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', is_unbalance=True, n_jobs=5)

    # create parameter distribution for parameter tuning
    param_dist = {
        'learning_rate': [0.005, 0.01, 0.1],
        'n_estimators': randint(50,300),
        'num_leaves': randint(20, 80),
        'feature_fraction':[0.5, 0.6, 0.7, 0.8],
        'bagging_fraction':[0.5, 0.6,0.7,0.8],
        'bagging_freq': randint(10,20)
    }

    # return model dict
    return {'name':"GBDT", 'model':gbm, 'param_dist':param_dist, 'n_iter': 20}

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __name__ == '__main__':
    # load preprocessed training dataset
    train = pd.read_csv('Data/train_processed.csv')

    # specify predictors and target columns
    target = "Survived"
    predictors =  [x for x in train.columns if x not in [target]]

    # fit model with random parameter search
    model = getModel()
    random_search = RandomizedSearchCV(model['model'], param_distributions=model['param_dist'], n_iter=model['n_iter'])
    random_search.fit(train[predictors], train[target])

    # Print top 5 scores and related param options
    results = random_search.cv_results_
    for i in range(1, 6):
        scores = np.flatnonzero(results['rank_test_score'] == i)
        for score in scores:
            print("Rank: {0}".format(i))
            print("score - mean: {0:.3f}, std: {1:.3f}".format(
                  results['mean_test_score'][score],
                  results['std_test_score'][score]))
            print("Parameters: {0}".format(results['params'][score]))

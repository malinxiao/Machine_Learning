from sklearn.linear_model import LogisticRegression
from scipy.stats import randint

def getModel():
    # create logistic regression classifier
    lr = LogisticRegression()

    # create parameter distribution for parameter tuning
    param_dist = {'penalty': ['l1','l2'], 
                  'C': [0.001,0.01,0.1,1,10,100,1000]}

    # return model dict
    return {'name':"Logistic Regression", 'model':lr, 'param_dist':param_dist, 'n_iter': 10}

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __name__ == '__main__':
    train = pd.read_csv('Data/train_processed.csv')

    target = "Survived"
    predictors =  [x for x in train.columns if x not in [target]]

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
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

def getModel():
    # create random forest classifer
    rf = RandomForestClassifier()

    # create parameter distribution for parameter tuning
    param_dist = {"max_depth": randint(6,9),
                  "max_features": ['auto', 12],
                  'n_estimators': [20, 50, 100, 150, 200, 250],
                  "min_samples_split": randint(2, 10),
                  "min_samples_leaf": randint(2, 8),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # return model dict
    return {'name':"Random Forest", 'model':rf, 'param_dist':param_dist, 'n_iter': 20}

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
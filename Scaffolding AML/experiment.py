import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

from azureml.logging import get_azureml_logger
run_logger = get_azureml_logger()

import model_GBDT
import model_lr
import model_RF 

def runExperiment():
    # load preprocessed training dataset
    train = pd.read_csv('Data/train_processed.csv')

    # specify predictors and target columns
    target = "Survived"
    predictors =  [x for x in train.columns if x not in [target]]

    # get models from model files
    models = [model_GBDT.getModel(), model_lr.getModel(), model_RF.getModel()]

    # fit models with random parameter search and log the best score for each model to AML job run dashboard
    for model in models:
        random_search = RandomizedSearchCV(model['model'], param_distributions=model['param_dist'], n_iter=model['n_iter'])
        random_search.fit(train[predictors], train[target])
        results = random_search.cv_results_
        scores = np.flatnonzero(results['rank_test_score'] == 1)
        score = results['mean_test_score'][scores[0]]
        run_logger.log(model['name'], round(score, 3))


if __name__ == '__main__':
    runExperiment()

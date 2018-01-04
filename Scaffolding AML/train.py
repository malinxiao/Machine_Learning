import pickle
import sys
import os

import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

# load processed data
train = pd.read_csv('Data/train_processed.csv')

target = "Survived"
predictors =  [x for x in train.columns if x not in [target]]

# train model
rf = RandomForestClassifier(max_features=12,\
                            criterion="entropy",\
                            min_samples_split=4,\
                            bootstrap=True,\
                            min_samples_leaf=2,\
                            max_depth=6,\
                            n_estimators=200)
rf.fit(train[predictors], train[target])

# serialize the model on disk in the special 'outputs' folder
f = open('./outputs/model.pkl', 'wb')
pickle.dump(rf, f)
f.close()

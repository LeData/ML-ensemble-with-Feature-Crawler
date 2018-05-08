import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
import gc
import fraud_feature_engineering
import fraud_models


local_path='input/'
current_skip=144038095
#options are 'test','train' and 'train sample', with nrows argument:

'''#CatBoost model
params = {'depth': 5,
 'iterations': 100,
 'l2_leaf_reg': 80,
 'learning_rate': .01,
 'custom_metric':'AUC',
 'random_seed': 714,
 'scale_pos_weight':640}

CatClf=CatClassifier('ante_day','last_day',params)
CatClf.reduce_train(2*10**7,3*10**7)
CatClf.pre_process('train')
CatClf.fit_model()'''

#LightGBM model
lgb_params2 = {
    'num_leaves': 2**5 - 1,
    'objective': 'regression_l2',
    'max_depth': 8,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 4,
    'num_boost_round': 1000,
    'early_stopping_rounds': 50
}

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.10,
    # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 200,  # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': -1,
    'verbose': 0,
}
#xgb model
xgb_params = {
    'depth': 4,
    'custom_metric':'AUC',
    'objective':'binary:logistic',
    'learning_rate':.0054,
    'scale_pos_weight':640,
    'seed':714
}

model=lgbmClassifier('ante_day',lgb_params)
model.pre_process('train')
model.fit_model()
model.feature_importance()
model.get_submission()

#kaggle competitions submit [-h] -c COMPETITION -f FILE -m MESSAGE

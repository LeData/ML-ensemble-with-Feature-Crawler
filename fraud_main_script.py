import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
import gc
from fraud_models import *


local_path='input/'
seed=714
#options are 'test','train' and 'train sample', with nrows argument:

# CatBoost model
cat_params = {
 'depth': 5,
 'iterations': 100,
 'l2_leaf_reg': 80,
 'learning_rate': .01,
 'custom_metric':'AUC',
 'random_seed': 714,
 'scale_pos_weight':640}

# LightGBM model
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
    'early_stopping_rounds': 50}

# Xgb model
xgb_params = {
    'depth': 4,
    'custom_metric':'AUC',
    'objective':'binary:logistic',
    'learning_rate':.0054,
    'scale_pos_weight':640,
    'seed':714}

# level -1 (optional):
# add new features to the available pool of features

# level 0:
sample_index = get_training_sample() # make subsampled set of the training set

# Level 1
# train 5 models of each engine
# !! Do not prune the crawler's trees !!
level1=load_data(train=False)
lgbm_model=lgbmClassifier(sample_index,lgbm_params)
xgb_model=XGBoostClassifier(sample_index,xgb_params)
cat_model=CatClassifier(sample_index,cat_params)

for i in range(5):
    features=select_node()
    level1['lgbm{}'.format(i+1)]=lgbm_model.fit_predict()
    level1['xgb{}'.format(i+1)]=xgb_model.fit_predict()
    level1['cat{}'.format(i+1)]=cat_model.fit_predict()

# level 2:
# make frame with predictions from level 1 models on the subsampled set
# train 1 lgbm model
# train 1 xgb model
# train 1 catboost model

# level 3
# make weighted average of level 2 predictions


#kaggle competitions submit [-h] -c COMPETITION -f FILE -m MESSAGE

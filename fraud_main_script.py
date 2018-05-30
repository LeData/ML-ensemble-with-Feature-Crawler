import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import yaml
import os
import gc
from fraud_models import *


local_path='input/'
config_path='config/'
seed=714
#options are 'test','train' and 'train sample', with nrows argument:

# CatBoost model
cat_params = {
 'depth': 5,
 'iterations': 100,
 'l2_leaf_reg': 80,
 'learning_rate': .01,
 'custom_metric':'AUC',
 'random_seed': seed,
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
    'seed':seed}

model_dict={'lgbm':lgb_params}
with open(local_path+'level1models.yaml','w') as File:
    yaml.dump(model_dict,File)

# level 0 (optional):
# add new features to the config file of the faeture manager of level1

# Level 1
level1-LevelOne(model_dict,local_path,config_path)
stopping_cond={'number':1,'threshold':0.85}
level1.learn_until(stopping_cond)
level2=LevelTwo(level1.get_level2_data(),model_list)
level2.train()
level3=LevelThree(level2.get_level3_data())
level3.get_submission()
level3.submit_to_kaggle()
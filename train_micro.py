
# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
from itertools import groupby
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
import read_data as rd
import numpy as np
def group_query_id (query_ids):
    return [len(list(group)) for key, group in groupby(query_ids)]

# load or create your dataset
print('Load data...')
df_train = pd.read_csv('canvas_data/train.csv',nrows=10000,header=0)
df_test = pd.read_csv('canvas_data/test.csv',nrows=20000,header=0)
df_eval = pd.read_csv('canvas_data/train.csv',skiprows=range(1, 10000),nrows=1000,header=0)

train_data = rd.aggregate_comp(df_train)
train_data = rd.filter_dates(train_data)
label_train = train_data['position']
train_data.drop('position', axis=1)
train_group = group_query_id(train_data['srch_id'])
# train_data.pop('srch_id')

eval_data = rd.aggregate_comp(df_eval)
eval_data = rd.filter_dates(eval_data)
label_eval = eval_data['position']
eval_data.drop('position', axis=1)
eval_group = group_query_id(eval_data['srch_id'])
# eval_data.pop('srch_id')

test_data = rd.aggregate_comp(df_test)
test_data = rd.filter_dates(test_data)
test_group = group_query_id(test_data['srch_id'])
# test_data.pop('srch_id')

# create dataset for lightgbm
lgb_train = lgb.Dataset(train_data, label=label_train, group=train_group)
lgb_eval = lgb.Dataset(eval_data, label=label_eval, group=eval_group, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': 1,
    'label_gain': '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40'
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
# eval
# print('The rmse of prediction is:', metrics.auc(test_data, y_pred))
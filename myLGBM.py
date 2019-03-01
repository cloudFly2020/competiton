import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error,make_scorer
import lightgbm as lgb

import os
import time
import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')
#from gen_train_feat2 import gen_train_data

train_file='temp/trainData.pkl'

def baseline():
    baseline_file='temp/baseline.pkl'
    if os.path.exists(baseline_file):
        baseline=pickle.load(open(baseline_file,'rb'))
    else:
        train_data=pickle.load(open(train_file,'rb'))#要open

        test_data=pickle.load(open('feat/testData.pkl'),'rb')
        test_data.fillna(0,inplace=True)
        sub_data =test_data['uid'].copy()
        test_data=test_data.values
        sub_pred=[]
        del test_data['uid']

        train_data.fillna(0.,inplace=True)
        train_data.drop('day',axis=1,inplace=True)#一定要删除
        #print(train_data.day)
        print(train_data.head())

        #label=train_data['label']
        label=train_data['label'].values# nd.array格式

        feature_list=list(train_data.columns)
        feature_list.remove('uid')#list才有remove
        feature_list.remove('label')
        #training=train_data[feature_list]
        training=train_data[feature_list].values

    #划分训练集和测试集
        #lgb_model.py
        kf=KFold(n_splits=5,random_state=2019,shuffle=True)
        #predictList=[]
        rmseList=[]

        #参数是否放循环里
        param = {
            'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression',
            'metric': {'l2', 'rmse'}, 'max_depth': 5, 'num_leaves': 21,
            'min_data_in_leaf': 300, 'learning_rate': 0.02,
            'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 5,
            'verbose': -1, 'num_boost_round': 2000
        }
        param2 = {'num_leaves': 120,#0.7869
                 'min_data_in_leaf': 30,
                 'objective': 'regression',
                 'max_depth': -1,
                 'learning_rate': 0.02,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.75,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.75,
                 "bagging_seed": 11,
                 "metric": 'rmse',
                 "lambda_l1": 0.1,
                 "verbosity": -1}
        # 得到训练集 验证集索引 训练集和验证集都有train和label
        for train_index,validata_index in kf.split(training,label):
            X_train,y_train,X_validata,y_validata=training[train_index],label[train_index],training[validata_index],label[validata_index]
            lgb_train=lgb.Dataset(X_train,label=y_train,feature_name=feature_list)
            lgb_val=lgb.Dataset(X_validata,label=y_validata)



            lgb_model=lgb.train(params=param,train_set=lgb_train,
                             num_boost_round=2000,valid_sets=lgb_val,early_stopping_rounds=100,verbose_eval=200)#verbose_eval训练显示条数
        #for fold_, (train_index, validata_index) in enumerate(kf.split(training,label)):
            #print('kfold',fold_+1)

            # TypeError: float() argument must be a string or a number, not 'Timedelta' 数据中有 days
            # lgb_train=lgb.Dataset(training[train_index],label[train_index])
            # lgb_val=lgb.Dataset(training[validata_index],label[validata_index])
            # lgb_model=lgb.train(params=param,train_set=lgb_train,

            # lgb_model=lgb.train(param,train_set=lgb_train,
            #                     valid_sets=lgb_val,
            #                     verbose_eval=200,
            #                     early_stopping_rounds=100)
            #num_round=20000
            #lgb_model =lgb.train(param,lgb_train, num_round, valid_sets=[lgb_train, lgb_val], verbose_eval=200, early_stopping_rounds=100)
            #如果迭代100次rmse还不下降 提前终止训练不训练1500
            #lgb_predict=lgb_model.predict(training[validata_index],num_iteration=lgb_model.best_iteration)
            #print("均方误差",mean_squared_error(label[validata_index],lgb_predict)**0.5)#平方根均方误差 题目要求
            lgb_predict=lgb_model.predict(X_validata,num_iteration=lgb_model.best_iteration)

            #最后提交 用 test数据(9,10,11)预测 预测结果为12月
            rmse=mean_squared_error(y_validata,lgb_predict)**0.5#平方根均方误差 题目要求
            print("均方误差",rmse)#平方根均方误差 题目要求
            test_predict=lgb_model.predict(test_data,num_iteration=lgb_model.best_iteration)
            #predictList.append(lgb_predict)
            rmseList.append(rmse)
            sub_pred.append(test_predict)
        #print("rmse",np.mean(predictList))
        #print("rmse",np.mean(np.array(predictList),axis=0)) 不是放预测值
        print("rmselist:{}\n rmse mean:{}".format(rmseList,np.mean(np.array(rmseList))))
        baseline=lgb_model
        #pickle.dump(baseline,open(baseline_file,'wb'))
        # #rmse:1.7849059799293876 加order click rmse mean:1.7795464662534068
    pred=np.mean(np.array(sub_pred))
    sub_data[:,'pred']=pred
    sub_data.to_csv("submission.csv",index=False,header=False,encoding='utf-8')
    return baseline



if __name__=='__main__':
    baseline=baseline()
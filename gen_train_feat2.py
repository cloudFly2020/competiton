# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from sklearn.externals import joblib# 模型保存
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle #数据序列化
import os
import math
from tool import *  #加载数据脱敏、日期转换函数库

t_click_file = 'data/t_click.csv'
t_loan_sum_file = 'data/t_loan_sum.csv'
t_loan_file = 'data/t_loan.csv'
t_order_file = 'data/t_order.csv'
t_user_file = 'data/t_user.csv'

if not os.path.exists('temp'):
    os.makedirs('temp')#创建文件夹


def gen_user_feat():
    """
    用户表处理
    1.判断处理后文件是否存在
    2.构造激活日期距离11-1多少周 然后删除激活日期属性 训练集只用11月前数据
    3.对初始额度反脱敏
    4.sex编码
    """
    dump_file='temp/train_user_feat.pkl'  #对t_user表数据处理保存
    if os.path.exists(dump_file):
        #pickle.load(dump_file)
        user_feat=pickle.load(open(dump_file,'rb'))
    else:
        user=pd.read_csv(t_user_file,header=0)
        print(user.columns)
        user['day']=user['active_date'].map(lambda x:datetime.strptime('2016-11-1','%Y-%m-%d')-datetime.strptime(x,'%Y-%m-%d'))
        user['week']=user['day'].map(lambda x:round(x.days/7))
        del user['active_date']
        user['limit']=user['limit'].map(lambda x:change_data(x))
        user_sex=pd.get_dummies(user['sex'],prefix='sex')
        user_feat=pd.concat([user,user_sex],axis=1)#concat直接拼接不需要关联属性 merge要有关联属性
        del user_feat['sex']
        pickle.dump(user_feat,open(dump_file,'wb'))
    return user_feat

def gen_loan_feat():
    loan_file='temp/train_loan_feat.pkl'
    if os.path.exists(loan_file):
        #pickle.load(loan_file)
        loan=pickle.load(open(loan_file,'rb'))
    else:
        loan=pd.read_csv(t_loan_file,header=0)
        print(loan.head())
        print("loan表初始的行列数：", loan.shape)
        loan['month']=loan['loan_time'].map(lambda x:get_month(x))
        loan=loan[loan['month']!=11]#训练集排除11月份数据
        loan['loan_amount']=loan['loan_amount'].map(lambda x:change_data(x))
        print("loan表的行列数：", loan.values.shape)

        print("---------------")
        #贷款时间特征 按比例 不直接用次数
        loan_hour=loan.copy()

        #print(loan_hour.head())
        #loan_hour['loan_hour']=loan_hour['loan_time'].map(lambda x:int(x.split(' ')[1].split(':')[0]))
        loan_hour['loan_hour']=loan_hour['loan_time'].map(lambda x:get_hour(x))
        #下面把24时间映射到6个区间
        loan_hour['loan_hour']=loan_hour['loan_hour'].map(lambda x: hour2bucket('loan',x))
        #print(loan_hour['loan_hour'].unique())
        #统计每一个用户不同时间段贷款次数
        loan_hour=loan_hour.groupby(['uid','loan_hour'],as_index=False).count()#用次数不好 用百分比更好
        #.reset_index  写不写没干系 但是 as_index=False 不能少
        loan_hour=loan_hour.pivot(index='uid',columns='loan_hour',values='loan_amount').reset_index()
        print("loan_hour",loan_hour.columns)
        loan_hour=loan_hour.fillna(0)
        loan_hour['loan_hour_sum']=loan_hour[['loan_hour_01','loan_hour_02','loan_hour_03','loan_hour_04','loan_hour_05',\
            'loan_hour_06']].apply(lambda x:x.sum(),axis=1)#不能用map
        loan_hour['loan_hour_01']=loan_hour['loan_hour_01']/loan_hour['loan_hour_sum']
        loan_hour['loan_hour_02']=loan_hour['loan_hour_02']/loan_hour['loan_hour_sum']
        loan_hour['loan_hour_03']=loan_hour['loan_hour_03']/loan_hour['loan_hour_sum']
        loan_hour['loan_hour_04']=loan_hour['loan_hour_04']/loan_hour['loan_hour_sum']
        loan_hour['loan_hour_05']=loan_hour['loan_hour_05']/loan_hour['loan_hour_sum']
        loan_hour['loan_hour_06']=loan_hour['loan_hour_06']/loan_hour['loan_hour_sum']
        loan_hour.drop('loan_hour_sum',axis=1,inplace=True)#axis一定写
        print("下面贷款月份统计特征")
        #贷款月份统计特征                   #离散用占比 连续用最大最小平均方差中位数
        loan_static=loan.copy()
        loan_static=loan_static.groupby(['uid','month'],as_index=False).sum()
        #简单写法
        stat_feat=['min','max','median','std','mean']
        loan_static=loan_static.groupby(['uid'])['loan_amount'].agg(stat_feat).reset_index()#不加reset_index uid成索引
        loan_static.columns=['uid']+['month_'+col for col in stat_feat]
        print('stat_feat',loan_static.columns)
        # loan_static=loan_static.pivot(index='uid',columns='month',values='loan_amount').reset_index()
        # loan_static=loan_static.fillna(0)
        #已经是透视表列是月份 axis=1 横向   columns='month' 8,9,10是整数
        #loan_static['month_min']=loan_static[['8','9','10']].apply(lambda x:x.min(),axis=1)
        # loan_static['month_min']=loan_static[[8,9,10]].apply(lambda x:x.min(),axis=1)
        # loan_static['month_max']=loan_static[[8,9,10]].apply(lambda x:x.max(),axis=1)
        # loan_static['month_mean']=loan_static[[8,9,10]].apply(lambda x:x.mean(),axis=1)
        # loan_static['month_sum']=loan_static[[8,9,10]].apply(lambda x:x.sum(),axis=1)
        # loan_static['month_median']=loan_static[[8,9,10]].apply(lambda x:x.median(),axis=1)
        # loan_static['month_std']=loan_static[[8,9,10]].apply(lambda x:x.std(),axis=1)

        #贷款分期 特征  count()? 也用占比好
        loan_plannum=loan.copy()
        loan_plannum=loan_plannum.groupby(['uid','plannum'],as_index=False).count()
        print('groupby',loan_plannum.head())
        loan_plannum = loan_plannum.pivot(index='uid',columns='plannum',values='loan_amount').reset_index()
        loan_plannum.fillna(0,inplace=True)
        print('loan_plannum',loan_plannum.columns)
        loan_plannum.columns=['uid','plannum_01','plannum_03','plannum_06','plannum_12']
        # loan_plannum['plannum_min']=loan_plannum[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x:x.min(),axis=1)
        # loan_plannum['plannum_max']=loan_plannum[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x:x.max(),axis=1)
        # loan_plannum['plannum_mean']=loan_plannum[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x:x.mean(),axis=1)
        # loan_plannum['plannum_median']=loan_plannum[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x:x.median(),axis=1)
        # loan_plannum['plannum_std']=loan_plannum[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x:x.std(),axis=1)

        loan_plannum['plannum_sum']=loan_plannum[['plannum_01','plannum_03','plannum_06','plannum_12']].apply(lambda x:x.sum(),axis=1)
        loan_plannum['plannum_01']=loan_plannum['plannum_01']/loan_plannum['plannum_sum']

        loan_plannum['plannum_03']=loan_plannum['plannum_03']/loan_plannum['plannum_sum']
        loan_plannum['plannum_06']=loan_plannum['plannum_06']/loan_plannum['plannum_sum']
        loan_plannum['plannum_12']=loan_plannum['plannum_12']/loan_plannum['plannum_sum']

        #每月贷款次数特征
        #1.上次贷款距离现在时间 算出贷款权重
        loan['distanceTimeWeight']=loan['loan_time'].map(lambda x:datetime.strptime('2016-11-1 00:00:00','%Y-%m-%d %H:%M:%S')-datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        #loan['distanceTimeWeight']=loan['distanceTimeWeight'].map(lambda x:round(1/(1+x.days/7)))
        loan['distanceTimeWeight']=loan['distanceTimeWeight'].map(lambda x:1/(1+round(x.days/7)))
        #贷款权重
        loan['loan_weight']=loan['loan_amount']*loan['distanceTimeWeight']


        # 新加 最后一次贷款距离11月时间 平均贷款间隔 最后一次贷款距离11月时间是否超过平均贷款间隔
        # 最后一次贷款 通过按uid和loan_time排序后去重取得
        loan_time=loan.copy()
        print('loan_time',loan_time.columns)
        loan_time=loan_time.sort_values(by=['uid','loan_time'])#'loan_time'无
        loan_time['loan_time']=loan_time['loan_time'].map(lambda x:datetime.strptime('2016-11-1 0:00:00','%Y-%m-%d %H:%M:%S')-datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        loan_time['diff_week']=loan_time['loan_time'].map(lambda x:round(x.days/7))
        last_loan_time=loan_time.copy()
        last_loan_time.drop_duplicates(subset='uid',keep='last',inplace=True)

        #last_loan_time.drop_duplicates(subset='uid', keep='last', inplace=True)
        print('last_loan_time',last_loan_time.head())
        last_loan_time=last_loan_time[['uid','diff_week']]
        last_loan_time['per_loan_time_interval']=13#默认13周 3个月
        #for id in list(last_loan_time.uid):
        for id in list(last_loan_time.index):
            uid = last_loan_time.loc[id,'uid']
            #interval=np.array(loan_time[loan_time['uid']==id]['diff_week'])
            interval = np.array(loan_time[loan_time['uid'] == uid]['diff_week'])
            if len(interval)==1:
                #last_loan_time['per_loan_interval']=13-interval[0]#这样不知道uid是谁
                #last_loan_time.loc[id,'per_loan_interval']=13-interval[0]
                per_loan_time_interval=13-interval[0]
            else:
                #last_loan_time.loc[id,'per_loan_interval']=(interval.max()-interval.min())/(len(interval)-1)
                per_loan_time_interval=(interval.max()-interval.min())/(len(interval)-1)
             #维度不一样last_loan_time  loan_time
            last_loan_time.loc[id,'per_loan_time_interval'] = per_loan_time_interval
        #整体对应减
        last_loan_time['exceed_per_interval']=last_loan_time['diff_week']-last_loan_time['per_loan_time_interval']
        print("last_loan_time",last_loan_time.columns)


        #每月贷款是否超过初始额度 先获取user表处理后保存的数据
        loan_exceed = loan.copy()
        loan_exceed = loan_exceed.groupby(['uid', 'month'], as_index=False).sum().reset_index()
        loan_exceed = loan_exceed.pivot(index='uid', columns='month', values='loan_amount').reset_index()
        loan_exceed = loan_exceed.fillna(0)

        user_limit=gen_user_feat()[['uid','limit']]#只需要两个特征数据 #file must have 'read' and 'readline' attributes删除文件
        loan_exceed=pd.merge(loan_exceed,user_limit,how='left',on='uid')
        #loan['loan_exceed']=loan['loan_amount']-loan['limit'] 平均是否超额 不好
        # def is_exceed(x):
        #     if x > 0:
        #         return 1
        #     else:
        #         return 0
        # loan['loan_exceed']=loan['per_month_amount']-loan['limit']
        # loan['loan_exceed']=loan['loan_exceed'].map(lambda x:is_exceed(x))
        loan_exceed['loan_exceed8'] = loan_exceed[8] - loan_exceed['limit']
        loan_exceed['loan_exceed9'] = loan_exceed[9] - loan_exceed['limit']
        loan_exceed['loan_exceed10'] = loan_exceed[10] - loan_exceed['limit']

        #看每次贷款是否超额 如果超额说明提额了 添加新额度特征
        new_limit=loan.copy()
        new_limit=new_limit.groupby(['uid'])['loan_amount'].agg('max').reset_index()
        new_limit.columns=['uid','max']
        new_limit=pd.merge(new_limit,user_limit,how='left',on='uid')
        limit=[]#最后拼接
        for i in range(len(new_limit['limit'])):
            #if new_limit.loc[i,'limit']>new_limit.loc[i,'loan_amount']:
            if new_limit.loc[i, 'limit'] > new_limit.loc[i, 'max']:
                limit.append(new_limit['limit'][i])
            else:
                #new_limit['new_limit']=new_limit['max'][i]
                limit.append(new_limit['max'][i])
        new_limit['new_limit']=limit
        #del loan['limit']

        #激活时间和首次贷款时间间隔
        loan_active=gen_user_feat()[['uid','week']]
        #time_df=loan.drop_duplicates(subset='loan_time',keep='first',inplace=True)
        #前面有loan_time  时间是距离2016-11-1周
        loan_time.drop_duplicates(subset='uid',keep='first',inplace=True)
        first_loan=pd.merge(loan_time,loan_active,how='left',on='uid')
        first_loan['active_loan_distance']=first_loan['week']-first_loan['diff_week']
        first_loan=first_loan[['uid','active_loan_distance']]
        print('11月份累加要还款的金额')
        # 2.11月份累加要还款的金额
        # month8 = loan[loan['month'] == 8].copy()  # 最好加copy
        # month8['repay'] = month8['loan_amount'] / month8['plannum']
        # # month8[month8['plannum']<3].loc[:,'repay']=0
        # month8.loc[month8['plannum'] < 3, 'repay'] = 0
        # month9 = loan[loan['month'] == 9].copy()
        # month9['repay'] = month9['loan_amount'] / month9['plannum']
        # month9.loc[month9['plannum'] < 2, 'repay'] = 0
        # month10 = loan[loan['month'] == 10].copy()
        # month10['repay'] = month10['loan_amount'] / month10['plannum']
        # loan = pd.concat([month8, month9, month10], axis=0, ignore_index=False)  # 0 无新特征
        # print('loan month8910', loan.head())
        # month_loan = pd.get_dummies(loan['month'], prefix='month')
        # loan = pd.concat([loan, month_loan], axis=1)  # 0是行纵向拼接无新特征 1是列横向拼接有新特征
        # print('loan get_dumies', loan.head())
        # loan=loan.groupby(['uid'],as_index=False).sum()
        #上面太繁琐 可以先整体算 再根据条件替换0
        loan['repay']=loan['loan_amount']/loan['plannum']
        for id in list(loan.index):
            if loan.loc[id,'plannum']==1 and loan.loc[id,'month']<=9:
                loan.loc[id,'repay']=0
        loan=loan.groupby(['uid','month'],as_index=False).sum()

        # 3.每月贷款间隔 是否有连续贷款情况
        #loan_month=loan.copy()
        loan_month=pd.get_dummies(loan['month'],prefix='month')  #无相同特征
        #loan=pd.merge(loan,loan_month,how='left',on='uid')
        #loan = pd.concat(loan,loan_month, axis=1)
        loan = pd.concat([loan, loan_month], axis=1)
        loan=loan.groupby(['uid'],as_index=False).sum()
        loan['loan_sum_month8910'] = loan['month_8'] + loan['month_9'] + loan['month_10']  # 这几个特征get_dummies
        loan['loan_continue89'] = loan['month_8'] + loan['month_9']
        loan['loan_continue89'] = loan['loan_continue89'].map({0: 1, 1: 0, 2: 1})  # 是否连续贷款做映射
        loan['loan_continue810'] = loan['month_8'] + loan['month_10']
        loan['loan_continue810'] = loan['loan_continue810'].map({0: 1, 1: 0, 2: 1})
        loan['loan_continue910'] = loan['month_9'] + loan['month_10']
        loan['loan_continue910'] = loan['loan_continue910'].map({0: 1, 1: 0, 2: 1})
        loan['loan_continue_month8910'] = loan['month_8'] + loan['month_9'] + loan['month_10']
        loan['loan_continue_month8910'] = loan['loan_continue_month8910'].map({0: 1, 1: 0, 2: 0, 3: 1})

        # 平均每月分期期数 平均每月贷款金额
        loan['per_month_plannum'] = loan['plannum'] / loan['loan_sum_month8910']
        loan['per_month_amount'] = loan['loan_amount'] / loan['loan_sum_month8910']

        del loan['month']
        del loan['month_8']
        del loan['month_9']
        del loan['month_10']
        #最后拼接
        loan=pd.merge(loan,loan_static[['uid','month_min','month_mean','month_max','month_std','month_median']],how='outer',on='uid')
        loan=pd.merge(loan,loan_hour,how='left',on='uid')
        loan=pd.merge(loan,last_loan_time,how='left',on='uid')
        loan=pd.merge(loan,new_limit[['uid','new_limit']],how='left',on='uid')
        loan=pd.merge(loan,loan_exceed[['uid','loan_exceed8','loan_exceed9','loan_exceed10']],how='left',on='uid')
        loan=pd.merge(loan,first_loan,how='left',on='uid')
        loan=pd.merge(loan,loan_plannum,how='left',on='uid')
        loan.fillna(0,inplace=True)
        #pickle.dump(loan,'loan_feat.pkl')
        pickle.dump(loan,open(loan_file,'wb'))
        print('last col',loan.columns)
        #pickle.dump(df_loan,open(loan_file,'wb'))
    return loan
def loan_timeWindow_feat():
    loan_timeWindow_file="temp/train_loanTimeWindow_feat.pkl"
    if os.path.exists(loan_timeWindow_file):
        loan_timeWindow=pickle.load(open(loan_timeWindow_file,'rb'))
    else:
        loan_timeWindow=pd.read_csv(t_loan_file,header=0)
        loan_timeWindow['month']=loan_timeWindow['loan_time'].map(lambda x:get_month(x))
        loan_timeWindow['loan_amount']=loan_timeWindow['loan_amount'].map(lambda x:change_data(x))
        loan_timeWindow=loan_timeWindow[loan_timeWindow['month']!=11]

        loan_timeWindow['interval_day']=loan_timeWindow['loan_time'].map(lambda x:datetime.strptime("2016-11-1 0:00:00","%Y-%m-%d %H:%M:%S")-datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
        loan_timeWindow['interval_day']=loan_timeWindow['interval_day'].map(lambda x:x.days)

        # 贷款行为在滑动时间窗口内的贷款统计特征
        print("构造dataframe")
        uid=loan_timeWindow['uid'].unique()
        temp=[1]*len(uid)#全为1
        loan_window=pd.DataFrame({'uid':uid,'temp':temp})
        #距离2016-11-1 不同时间间隔贷款金额统计特征
        window_list=[0,3,7,14,21,28,35,42,49,56,63,70,77,84]
        for i in range(len(window_list)-1):
            dayBegin=window_list[i]
            dayEnd=window_list[i+1]
            data=loan_timeWindow[['uid','interval_day','loan_amount']].copy()
            window_feat=get_window_feat(data,'loan_amount','loan',dayBegin,dayEnd)
            loan_window=pd.merge(loan_window,window_feat,how='left',on='uid')
        loan_window=loan_window.fillna(0.)
        del loan_window['temp']
        loan_timeWindow=loan_window
        pickle.dump(loan_timeWindow,open(loan_timeWindow_file,'wb'))
    return loan_timeWindow#注意 return 位置

def gen_click_feat():
    click_file='temp/train_click.pkl'
    if os.path.exists(click_file):
        click_feat=pickle.load(open(click_file,'rb'))
    else:
        click_feat=pd.read_csv(t_click_file,header=0)
        #print(click_feat.head())
        click_feat['click_month']=click_feat['click_time'].map(lambda x:get_month(x))
        click_feat=click_feat[click_feat['click_month']!=11]

        #特征提取和loan表类似 各时间段点击占比 点击参数占比
        click_hour=click_feat.copy()
        click_hour['click_hour']=click_hour['click_time'].map(lambda x:get_hour(x))
        click_hour['click_hour']=click_hour['click_hour'].map(lambda x:hour2bucket('click',x))
        #TypeError: '>=' not supported between instances of 'str' and 'int' 没取小时或反了
        #print(click_hour.head())
        click_hour=click_hour.groupby(['uid','click_hour'],as_index=False).count()
        click_hour=click_hour.pivot(index='uid',columns='click_hour',values='click_time').reset_index()
        click_hour.fillna(0,inplace=True)
        print(click_hour.head())
        # click_hour['hour_sum_click']=click_hour[['click_hour_01','click_hour_02','click_hour_03',\
        #                                          'click_hour_04','click_hour_05','click_hour_06']].sum(axis=1)
        click_hour['hour_sum_click']=click_hour[['click_hour_01','click_hour_02','click_hour_03',\
                                                'click_hour_04','click_hour_05','click_hour_06']].apply(lambda x:x.sum(),axis=1)
        click_hour['per_hour_click01']=click_hour['click_hour_01']/click_hour['hour_sum_click']
        click_hour['per_hour_click02']=click_hour['click_hour_02']/click_hour['hour_sum_click']
        click_hour['per_hour_click03']=click_hour['click_hour_03']/click_hour['hour_sum_click']
        click_hour['per_hour_click04']=click_hour['click_hour_04']/click_hour['hour_sum_click']
        click_hour['per_hour_click05']=click_hour['click_hour_05']/click_hour['hour_sum_click']
        click_hour['per_hour_click06']=click_hour['click_hour_06']/click_hour['hour_sum_click']
        click_hour.drop('hour_sum_click',axis=1,inplace=True)
        #三个月之内点击特征
        #一天记录就是一个点击
        click_feat['click']=1
        click_statics=click_feat.copy()
        statics_list=['max','min','mean','median','std','count','sum']
        #click_statics=click_statics.groupby(['uid','click_month'],as_index=False).agg(statics_list).reset_index()
        click_statics = click_statics.groupby(['uid', 'click_month'], as_index=False).sum()
        click_statics=click_statics.groupby(['uid'])['click'].agg(statics_list).reset_index()

        click_statics.columns=['uid']+['click_month_'+col for col in statics_list]
        click_statics.fillna(0,inplace=True)
        #print(click_statics.head())
        #最后一次点击距离11月时常 距离11平均间隔 最后一次点击是否超过平均间隔
        click_interval=click_feat.copy()
        click_interval=click_interval.sort_values(by=['uid','click_time'])
        print("click_interval",click_interval.head())
        click_interval['click_time']=click_interval['click_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        click_interval['last_click_interval']=click_interval['click_time'].map(lambda x:round(x.days/7))
        click_last=click_interval.copy()
        click_last.drop_duplicates(subset='uid',keep='last',inplace=True)
        click_last=click_last[['uid','last_click_interval']]
        click_last['per_click_interval']=13
        #下面算一个id多次点击距离11平均间隔
        click_interval=click_interval.groupby(['uid','last_click_interval'],as_index=False).sum()

        for idx in list(click_last.index):
            uid=click_last.loc[idx,'uid']
            #interval=list(click_interval[click_interval['uid']==uid]['last_click_interval'])
            interval=np.array(click_interval[click_interval['uid']==uid]['last_click_interval'])
            if len(interval)==1:
                per_interval=13-interval[0]
            else:
                per_interval=((interval.max()-interval.min())/len(interval)-1)
            click_last.loc[idx,'per_interval']=per_interval
        click_last['exceed_interval']=click_last['per_interval']-click_last['last_click_interval']
        print("click_last",click_last.head())
        click_feat['click_interval']=click_feat['click_time'].map(lambda x:datetime.strptime('2016-11-1 0:00:00','%Y-%m-%d %H:%M:%S')-datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        click_feat['interval_weight']=click_feat['click_interval'].map(lambda x:1/(1+round(x.days/7)))
        del click_feat['click_time']

        pid_data=pd.get_dummies(click_feat['pid'],prefix='pid')
        param_data=pd.get_dummies(click_feat['param'],prefix='param')

        click_feat.drop('pid',axis=1,inplace=True)
        click_feat.drop('param',axis=1,inplace=True)
        #click_feat=pd.concat(click_feat,pid_data,axis=1)
        click_feat=pd.concat([click_feat,pid_data,param_data],axis=1)
        #下面求pid param 次数占比
        #click_feat=click_feat.groupby(['uid','click'],as_index=False).sum()
        click_feat=click_feat.groupby(['uid'],as_index=False).sum()
        pid_list=list(pid_data.columns)
        param_list=list(param_data.columns)
        colums_list=pid_list+param_list
        for feat in colums_list:
            click_feat.loc[:,feat]=click_feat[feat]/(1+click_feat['click'])
        del click_feat['click_month']
        del click_feat['click']
        click_feat=pd.merge(click_feat,click_hour,how='left',on='uid')
        click_feat=pd.merge(click_feat,click_statics,how='left',on='uid')
        click_feat=pd.merge(click_feat,click_last,how='left',on='uid')
        #click_feat=pd.merge(click_feat,click_interval,how='left',on='uid')

        pickle.dump(click_feat,open(click_file,'wb'))
    return click_feat

def gen_order_feat():
    order_file='temp/train_order.pkl'
    if os.path.exists(order_file):
        order_feat=pickle.load(open(order_file,'rb'))
    else:
        order_feat=pd.read_csv(t_order_file,header=0)
        order_feat['month']=order_feat['buy_time'].map(lambda x:get_month(x))
        order_feat=order_feat[order_feat['month']!=11]
        #ValueError: cannot convert float NaN to integer
        order_feat.fillna(0,inplace=True)
        print(order_feat.head())
        order_feat['price']=order_feat['price'].map(lambda x:change_data(x))
        order_feat['discount']=order_feat['discount'].map(lambda x:change_data(x))
        # 购买商品实际支付
        order_feat['real_pay']=order_feat['price']*order_feat['qty']-order_feat['discount']
        order_feat.loc[order_feat['real_pay']<0,'real_pay']=0#有些真实价格负 价格进行z-score处理最小0 最大亿
        # 三个月的消费金额统计特征
        order_static=order_feat.copy()
        order_static=order_static.groupby(['uid','month'],as_index=False).sum()
        static_list=['min','max','mean','std','median','count']
        #for col in static_list: 不用循环 列表里循环
            #order_static['']=order_static[''] 不会写-_- 整体等于 groupby agg
        order_static=order_static.groupby(['uid'])['real_pay'].agg(static_list).reset_index()
        order_static.columns=['uid']+['real_pay_'+col for col in static_list]

        #最后一次购买距离11月份时间 平均间隔 最后一次距离11月时间是否超过平均间隔
        # order_time=order_feat.copy()
        # order_time=order_time.sort_values(by=['uid','buy_time'])
        # order_time['buy_time']=order_time['buy_time'].map(lambda x:datetime.strptime('2016-11-1','%Y-%m-%d')-datetime.strptime(x,'%Y-%m-%d'))
        # order_time['order_interval']=order_time['buy_time'].map(lambda x:round(x.days/7))
        # order_last_time=order_time.copy()
        # order_last_time.drop_duplicates(subset='uid',keep='last')
        # order_last_time=order_last_time[['uid','order_interval']]
        # order_last_time['per_interval']=13#默认
        # order_time=order_time.groupby(['uid','order_interval'],as_index=False).sum()#不能忘
        # print(order_time.shape)
        # print(order_last_time.head())
        # for id in list(order_last_time.index):
        #     uid=order_last_time.loc[id,'uid']
        #     #interval=np.array(order_time.loc[id,'interval'])
        #     interval=np.array(order_time.loc[order_time['uid']==uid,'order_interval'])
        #     if len(interval)==1:
        #         per_interval=13-interval[0]
        #     else:
        #         per_interval=(interval.max()-interval.min())/(len(interval)-1)
        #     order_last_time.loc[id,'per_interval']=per_interval
        # order_last_time['exceed_interval']=order_last_time['order_interval']-order_last_time['per_interval']
        # print('order_last_time',order_last_time.head())
        #购买权重
        order_feat['buy_weight']=order_feat['buy_time'].map(lambda x:datetime.strptime('2016-11-1','%Y-%m-%d')-datetime.strptime(x,'%Y-%m-%d'))
        order_feat['buy_weight'] =order_feat['buy_weight'].map(lambda x:1/(1+round(x.days/7)))

        order_feat['real_buy_weight']=order_feat['buy_weight']*order_feat['real_pay']
        order_feat=order_feat[['uid','buy_weight','real_buy_weight','discount','real_pay']]
        order_feat=order_feat.groupby(['uid'],as_index=False).sum()
        # 购买商品的折扣率
        order_feat['discount_ratio']=order_feat['discount']/order_feat['discount']+order_feat['real_pay']
        del order_feat['discount']

        order_feat=pd.merge(order_feat,order_static,how='left',on='uid')
        #order_feat=pd.merge(order_feat,order_last_time,how='left',on='uid')
        order_feat.fillna(0,inplace=True)#不要忘了

        pickle.dump(order_feat,open(order_file,'wb'))
    return order_feat

def get_label():
    """
    用11月数据做训练集
    :return:
    """
    loan_sum_11='temp/train_label.pkl'
    if os.path.exists(loan_sum_11):
        train_label=pickle.load(open(loan_sum_11,'rb'))
    else:
        train_label=pd.read_csv(t_loan_sum_file,header=0)
        train_label=train_label[['uid','loan_sum']]
        train_label.columns=['uid','label']
        pickle.dump(train_label,open(loan_sum_11,'wb'))
        print(train_label.describe())
    return train_label
def gen_train_data():
    train_file='temp/trainData.pkl'
    if os.path.exists(train_file):
        trainData=pickle.load(open(train_file,'rb'))
    else:
        user_data=gen_user_feat()
        loan_data=gen_loan_feat()
        loan_timeWindow=loan_timeWindow_feat()
        loan_order=gen_click_feat()
        loan_click=gen_click_feat()
        train_label=get_label()
        trainData=pd.merge(user_data,loan_data,how='left',on='uid')
        trainData=pd.merge(trainData,loan_order,how='left',on='uid')
        trainData=pd.merge(trainData,loan_click,how='left',on='uid')
        trainData=pd.merge(trainData,loan_timeWindow,how='left',on='uid')
        trainData=pd.merge(trainData,train_label,how='left',on='uid')

        pickle.dump(trainData,open(train_file,'wb'))
        feat_id={id:feat for id,feat in enumerate(list(trainData))}
        print(feat_id)
    return trainData

if __name__=='__main__':
    # user=gen_user_feat()
    # print(user.head())
    # print(user.shape)
    # loan = gen_loan_feat()
    # print(loan.head())
    # print(loan.shape)

    # loan_timeWindow = loan_timeWindow_feat()
    # print(loan_timeWindow.head())
    # print(loan_timeWindow.shape)
    # train_label = get_label()
    # print(train_label.head())
    # print(train_label.shape)
    train_data=gen_train_data()
    print(train_data.head())
    print(train_data.shape)

    # click=gen_click_feat()
    # print(click.head())
    # print(click.shape)
    # order=gen_order_feat()
    # print(order.head())
    # print(order.shape)
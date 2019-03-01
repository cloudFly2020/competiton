#encoding:utf-8
import numpy as np
import pandas as pd
import math
def change_data(x):
    return round(math.pow(5,x)-1)
def get_month(x):
    return int(x.split(' ')[0].split('-')[1])
def get_hour(x):
    return int(x.split(' ')[1].split(':')[0])
def hour2bucket(action, hour):
    #if hour>=8 & hour<=11:
    if hour>=8 and hour<=11:
        return '%s_hour_01' %action   #时间是整点
    elif hour>=12 and hour<=15:
        return '%s_hour_02' %action
    elif hour>=16 and hour<=19:
        return '%s_hour_03' % action
    elif hour >= 20 and hour <= 23:
        return '%s_hour_04' % action
    elif hour >= 4 and hour <= 7:
        return '%s_hour_05' % action
    else:
        return '%s_hour_06' % action

def get_window_feat(data,values,action,dayBegin,dayEnd):
    data=data[data['interval_day']>dayBegin]
    data=data[data['interval_day']<=dayEnd]
    stats=['min','max','mean','std','count','sum','median']
    data_feat=data.groupby('uid')[values].agg(stats).reset_index()
    data_feat.columns=['uid']+['%s_%s_'%(action,dayEnd)+col for col in stats]
    return data_feat
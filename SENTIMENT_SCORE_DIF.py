import pymysql
import pandas as pd
import datetime
from virgo import market 
import time
import datetime
from virgo import market as vm
from virgo import factor as vf
import numpy as np
import pandas as pd
##import numba as nb
##from numba import jit
import seaborn as sns
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import cycler
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator
sns.set()

def reformatDate(list):
    newlist=[]
    for i in list:
        i=str(i)
        b=i.replace('-','')
        newlist.append(b)
    return newlist

def date_range(beginDate, endDate):  #获取交易日和自然日
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y-%m-%d")
    return dates

#date_pool = market.trading_days("2018-06-01","2021-06-01")  #交易日
#date_pool_all = date_range("2018-06-01","2021-06-01")  #自然日

def checkDate(date):  #检验该日期是否有数据，返回boolean
    if os.access('/root/work/mfb/rr_data_all/' + date + '.pkl',os.F_OK):
        return True
    else:
        return False

def getDailyInfo(date):  #读取当个自然日的信息,不与交易日关联
    df = pd.read_pickle('/root/work/mfb/rr_data_all/' + date + '.pkl')
    df=df[['RPT_INSERT_TIME','SENTIMENT_CLASSIFY','SENTIMENT_SCORE','SENTIMENT_SCORE_AVG','TICKER_SYMBOL']]
    df['SENTIMENT_SCORE'] = pd.to_numeric(df['SENTIMENT_SCORE'],errors='coerce')  #这个变量的格式需要转换一下
    return df

def getMulti30_60dInfo(date):  #获得该交易日前30至60个自然日的信息，并与该交易日关联
    startDate = datetime.datetime.strptime(date,'%Y-%m-%d').date()  #字符串转化为date形式
    dflist = []
    count=0
    tmpDate=startDate+datetime.timedelta(-90)
    while count<90:
        tmpDate=tmpDate+datetime.timedelta(-1)  #往前取日期   
        if(checkDate(str(tmpDate))):
            dflist.append(getDailyInfo(str(tmpDate)))
            count+=1
    df=pd.concat(dflist)
    tmp = market.range_bars(date,date)
    tmp['TICKER_SYMBOL'] = tmp['symbol'].apply(lambda x: x[0:6])
    res = tmp[['TICKER_SYMBOL']]
    df = pd.merge(df,res,how = 'inner', on = 'TICKER_SYMBOL')
    df=df.groupby('TICKER_SYMBOL').agg('sum')
    return df

def getMulti30dSENTI_Dif(date):  #获得该交易日前30个自然日的信息，并与该交易日关联;每月末对个股该月的强相关新闻情绪值（去重）求平均，并与上个月作差
    startDate = datetime.datetime.strptime(date,'%Y-%m-%d').date()  #字符串转化为date形式
    dflist = []
    count=0
    tmpDate=startDate
    while count<90:
        tmpDate=tmpDate+datetime.timedelta(-1)  #往前取日期   
        if(checkDate(str(tmpDate))):
            dflist.append(getDailyInfo(str(tmpDate)))
            count+=1
    df=pd.concat(dflist)
    tmp = market.range_bars(date,date)
    tmp['TICKER_SYMBOL'] = tmp['symbol'].apply(lambda x: x[0:6])
    res = tmp[['TICKER_SYMBOL']]
    df = pd.merge(df,res,how = 'inner', on = 'TICKER_SYMBOL')
    df=df.groupby('TICKER_SYMBOL').agg('sum')
    df1=getMulti30_60dInfo(date)
    df = pd.merge(df,df1,how = 'inner', on = 'TICKER_SYMBOL')
    df['signal']=df['SENTIMENT_SCORE_x']-df['SENTIMENT_SCORE_y']
    df['date']=date
    df['symbol']=df.index
    df['date']=reformatDate(df['date'])
    df=df.reset_index()
    return df[['symbol','signal','date']]

def showMultiData(start,end):
    date_pool = market.trading_days(start,end)  #交易日
    df = pd.DataFrame()
    dflist = []
    for i in date_pool:
        dflist.append(getMulti30dSENTI_Dif(i))
        print(i) 
    df = pd.concat(dflist)
    return df

def outputSENTIMENT_SCORE_DIF(dataframe):
    dataframe.to_csv('/root/work/mfb/alpha/SENTIMENT_SCORE_DIF.csv')

outputSENTIMENT_SCORE_DIF(showMultiData('2018-12-03','2020-12-31'))


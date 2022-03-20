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
import math
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

def getDailyInfo(date):  #读取当个自然日的信息,不与交易日关联
    df = pd.read_pickle('/root/work/mfb/rr_data_all/' + date + '.pkl')
    df=df[['RPT_INSERT_TIME','SENTIMENT_CLASSIFY','SENTIMENT_SCORE','SENTIMENT_SCORE_AVG','TICKER_SYMBOL']]
    df['SENTIMENT_SCORE'] = pd.to_numeric(df['SENTIMENT_SCORE'],errors='coerce')  #这个变量的格式需要转换一下
    return df

def checkDate(date):  #检验该日期是否有数据，返回boolean
    if os.access('/root/work/mfb/rr_data_all/' + date + '.pkl',os.F_OK):
        return True
    else:
        return False

def getMulti30dInfo(date):  #获得该交易日前30个自然日的信息，并与该交易日关联
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
    return df

def SENTI_MONTHNUM_ADJ_formula(a,b,c,d):
    if((d%1==0)==False):
        if(d>4):
            SENTI_MONTHNUM_ADJ=math.floor(d)
        if(d<-8):
            SENTI_MONTHNUM_ADJ=math.ceil(d)
        if(-3<d<4):
            if(a>=(b+c)):
                SENTI_MONTHNUM_ADJ=math.ceil(d)
            else:
                SENTI_MONTHNUM_ADJ=math.floor(d)
        if(-8<d<-3):
            if((a+c)>=b):
                SENTI_MONTHNUM_ADJ=math.ceil(d)
            else:
                SENTI_MONTHNUM_ADJ=math.floor(d)
    else:
        SENTI_MONTHNUM_ADJ=d
        
    return SENTI_MONTHNUM_ADJ

def getMonthSENTI_NUM(date):
    df=getMulti30dInfo(date)
    df=df[['SENTIMENT_CLASSIFY','TICKER_SYMBOL']]
    dfpos=df[df['SENTIMENT_CLASSIFY']>0]
    dfneg=df[df['SENTIMENT_CLASSIFY']<0]
    dfneu=df[df['SENTIMENT_CLASSIFY']==0]
    dfpos=dfpos.groupby('TICKER_SYMBOL').agg('count')
    dfpos=dfpos.rename(columns={'SENTIMENT_CLASSIFY':'posCount'})
    dfneg=dfneg.groupby('TICKER_SYMBOL').agg('count')
    dfneg=dfneg.rename(columns={'SENTIMENT_CLASSIFY':'negCount'})
    dfneu=dfneu.groupby('TICKER_SYMBOL').agg('count')
    dfneu=dfneu.rename(columns={'SENTIMENT_CLASSIFY':'neuCount'})
    dfpos = pd.merge(dfpos,dfneg,how = 'outer', on = 'TICKER_SYMBOL')
    dfpos = pd.merge(dfpos,dfneu,how = 'outer', on = 'TICKER_SYMBOL')
    dfpos=dfpos.fillna(0)
    dfpos['SENTI_MONTHNUM']=4*dfpos['posCount']-8*dfpos['negCount']-3*dfpos['neuCount']/(dfpos['posCount']+dfpos['negCount']+dfpos['neuCount'])
    dfpos['date']=date
    dfpos['signal'] = dfpos.apply(lambda row: SENTI_MONTHNUM_ADJ_formula(row['posCount'], row['negCount'],row['neuCount'],row['SENTI_MONTHNUM']), axis=1)
    dfpos['symbol']=dfpos.index
    dfpos['date']=reformatDate(dfpos['date'])
    dfpos=dfpos.reset_index()
    return dfpos[['symbol','signal','date']]
def showMultiData(start,end):
    date_pool = market.trading_days(start,end)  #交易日
    df = pd.DataFrame()
    dflist = []
    for i in date_pool:
        dflist.append(getMonthSENTI_NUM(i))
        print(i) 
    df = pd.concat(dflist)
    return df


def outputSENTIMENT_SCORE_MonthSENTI_NUM(dataframe):
    dataframe.to_csv('/root/work/mfb/alpha/SENTIMENT_SCORE_MonthSENTI_NUM.csv')

outputSENTIMENT_SCORE_MonthSENTI_NUM(showMultiData('2018-09-04','2020-12-31'))


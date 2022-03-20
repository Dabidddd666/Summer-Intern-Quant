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

def getDailySENTI(date):  #读取当个自然日的信息，计算SENTI,不与交易日关联
    df = pd.read_pickle('/root/work/mfb/rr_data_all/' + date + '.pkl')
    df=df[['RPT_INSERT_TIME','HALF_POSITIVE_PROBABILITY','POSITIVE_PROBABILITY','NEGATIVE_PROBABILITY','NEUTRAL_PROBABILITY','SENTIMENT_SCORE','SENTIMENT_SCORE_AVG','TICKER_SYMBOL']]
    df['SENTIMENT_SCORE'] = pd.to_numeric(df['SENTIMENT_SCORE'],errors='coerce')  #这个变量的格式需要转换一下
    df['HALF_POSITIVE_PROBABILITY'] = pd.to_numeric(df['HALF_POSITIVE_PROBABILITY'],errors='coerce')  #这个变量的格式需要转换一下    
    df['POSITIVE_PROBABILITY'] = pd.to_numeric(df['POSITIVE_PROBABILITY'],errors='coerce')  #这个变量的格式需要转换一下    
    df['NEGATIVE_PROBABILITY'] = pd.to_numeric(df['NEGATIVE_PROBABILITY'],errors='coerce')  #这个变量的格式需要转换一下    
    df['NEUTRAL_PROBABILITY'] = pd.to_numeric(df['NEUTRAL_PROBABILITY'],errors='coerce')  #这个变量的格式需要转换一下
    df['dailySignal']=5*df['HALF_POSITIVE_PROBABILITY']+7*df['POSITIVE_PROBABILITY']+1*df['NEGATIVE_PROBABILITY']+3*df['NEUTRAL_PROBABILITY']
    df=df.groupby('TICKER_SYMBOL').agg('mean')  #若个股i在当日有多篇研报，则对多篇研报的情感得分再进行平均，得到c当日每只个股i的情感得分Scorei,c，若个股i在c当日无研报则情感得分为空。
    df['Symbol']=df.index
    df=df.reset_index()
    df['dailyDate']=date
    return df

def getMulti180Data(start):  #获取交易日前180天数据(自动剔除没有数据的日期)，作为该交易日的总数据集(包含已经算好的SENTI)，返回一个DF
    startDate = datetime.datetime.strptime(start,'%Y-%m-%d').date()  #字符串转化为date形式
    dflist = []
    count=0
    tmpDate=startDate
    while count<180:
        tmpDate=tmpDate+datetime.timedelta(-1)  #往前取日期   
        if(checkDate(str(tmpDate))):
            dflist.append(getDailySENTI(str(tmpDate)))
            count+=1
    df=pd.concat(dflist)
    tmp = market.range_bars(start,start)
    tmp['TICKER_SYMBOL'] = tmp['symbol'].apply(lambda x: x[0:6])
    res = tmp[['symbol','TICKER_SYMBOL']]
    df = pd.merge(df,res,how = 'inner', on = 'TICKER_SYMBOL')
    return df

def checkDate(date):  #检验该日期是否有数据，返回boolean
    if os.access('/root/work/mfb/rr_data_all/' + date + '.pkl',os.F_OK):
        return True
    else:
        return False

def getDailySENTI_ADJ(date):  ##计算交易日当天的SENTI_ADJ（取前九十天的数据）
    df=getMulti180Data(date)
    df=df[['dailySignal','Symbol','dailyDate']]
    df=pd.pivot_table(df,index=[u'Symbol'],columns=[u'dailyDate'])
    df=df.fillna(0)
    matrix=df.values
    df1=pd.DataFrame(columns=['weight'])
    i=df.shape[1]
    count=0
    while i>0:
        df1.loc[count]=1/i
        count+=1
        i-=1
    matrix1=df1.values
    SENTI_ADJ=np.dot(matrix,matrix1)
    df['SENTI_ADJ']=SENTI_ADJ
    df['date']=date
    df['symbol']=df.index
    df['date']=reformatDate(df['date'])
    df=df.rename(columns={'SENTI_ADJ':'signal'})
    df=df.reset_index()
    return df[['symbol','signal','date']]

def showMultiData(start,end):
    date_pool = market.trading_days(start,end)  #交易日
    df = pd.DataFrame()
    dflist = []
    for i in date_pool:
        dflist.append(getDailySENTI_ADJ(i))
        print(i) 
    df = pd.concat(dflist)
    return df

def outputSENTIMENT_SCORE_DAILYWEIGHTED_ADJ4(dataframe):
    dataframe.to_csv('/root/work/mfb/alpha/SENTIMENT_SCORE_DAILYWEIGHTED_ADJ4.csv')

outputSENTIMENT_SCORE_DAILYWEIGHTED_ADJ4(showMultiData('2018-12-08','2020-12-31'))


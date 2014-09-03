# -*- coding: utf-8 -*-
"""

This is a script to run on a portfolio or strategy ror series for basic research

Goal: consistent way of analyzing portfolio/strategy

@author: colin4567
"""
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pylab as plt
import pandas.io.data as web
import math
pd.set_option('notebook_repr_html', False)

import drawdown as dd


"""FUNCTIONS"""


def moments(df):
    ret = df.mean()
    vol = df.std() 
    skew = df.skew()
    excess_kurt = df.kurt() #already is excess
    dist = DataFrame({'ret':ret, 'vol':vol*math.sqrt(260), 'skew':skew, 'kurt':excess_kurt},
                     columns=['ret','vol', 'skew', 'kurt'])
    return dist.T


def semistandard_dist(df):
    neg = df[df < 0]
    neg_moments = moments(neg)
    return neg_moments.ix[['vol','skew','kurt']]


def ratios(df, bench):
    #sharpe ratio
    ret = df.mean()
    vol = df.std()
    ret2vol = np.round((ret/vol) * math.sqrt(260), 2)
    
    #treynor ratio = ann. return / beta - risk premium per unit of beta
    # pg. 142 CAIA book 2 - best used for well-diversified portfolios
    beta = df.corrwith(bench) * (df.std() / bench.std())
    treynor = np.round(((ret * math.sqrt(260))/beta) * math.sqrt(260), 2)
    
    #sortino ratio = portfolio return - necta return) / downside risk
    alpha = (df.sub(bench, axis=0)).mean()
    sortino = np.round((alpha/df[df < 0].std()) * math.sqrt(260), 2)
    
    #information ratio = port return - bench return / tracking error
    
    r = DataFrame({'sharpe (vol)': ret2vol, 'treynor (beta)': treynor, 'sortino (alpha/-vol)':sortino},
                  columns=['sharpe (vol)', 'treynor (beta)', 'sortino (alpha/-vol)'])
    return r.T



def regression(df, bench):
    return df.apply(lambda x: (pd.ols(y=x, x=bench).summary_as_matrix.x))
    
    

def correlations(df, bench):
    c = df.corrwith(bench)
    return c

    

def quant_var(df, c=.95): 
    #use with apply
    #significance = 1-confidence level     
    q = 1-c
    #number of observations required in the left tail
    n = int(math.ceil(q*df.count()))
    ranked_ror = df.order()
    #ascending by default
    return ranked_ror[n]


def quant_cvar(df, c=.99):
    #use with apply
    #UCIT compliant monthly 99% CVaR - avg of losses beyond VaR. Using daily here
    #df_m = df.resample('BM', how='sum')
    q = 1-c
    #number of observations required in the left tail
    n = int(math.ceil(q*df.count()))
    ranked_ror = df.order()
    #ascending by default
    return ranked_ror[:n].mean()
    
 
def return_index(ret):
    #use with apply
    ret.ix[0] = 1
    vami = ret.cumsum()
    return vami   


'''DATA'''


def get_px(rorStyle=0):
    getTicker = raw_input("Provide mutual fund tickers seperated by commas (no error catching here so be exact!): ").split(",")
    cleanTickers = [x.strip() for x in getTicker]
    print cleanTickers
    if rorStyle == 0:
        px = DataFrame({n: web.get_data_yahoo(n, start='1980-01-01')['Adj Close'].pct_change() for n in cleanTickers}).dropna()
    elif rorStyle == 1:
        px = np.log(DataFrame({n: web.get_data_yahoo(n, start='1980-01-01')['Adj Close'].pct_change() for n in cleanTickers}).dropna() + 1)
    return px

def get_bench(rorStyle=0):
    getTicker = raw_input("Provide mutual fund or index ticker to be used as benchmark (^gspc is sp500): ").split(",")
    cleanTicker = [x.strip() for x in getTicker]
    print "Benchmark is: ", cleanTicker
    if rorStyle == 0:
        ror = web.get_data_yahoo(cleanTicker, start='1980-01-01')['Adj Close'].pct_change().dropna()
    elif rorStyle == 1:
        px = web.get_data_yahoo(cleanTicker, start='1980-01-01')['Adj Close']
        ror = np.log(px / px.shift(1))
        
    return ror


def getRorStyle():
    while True:
        answer = raw_input("Do you want to compound ror's? (yes/no) ").lower()
        try: 
            answer in ['yes','y', 'no', 'n']
        except:
            print ("Try again, either yes/no or y/n")
            
        if (answer in ['yes','y']):
            return 1
        elif (answer in ['no', 'n']):
            return 0
                



    
def main(c=.99):

    retStyle = getRorStyle()
    data = get_px(retStyle)
    bench = get_bench(retStyle)
    dataBench = data.join(bench)
    benchSeries = bench.ix[:,0]
    dataBench.to_csv('test_ror.csv')    
    
    distribution = moments(dataBench)
    print "\n The distribution looks like: \n"    
    print distribution
    print "\n The downside risk from all neg returns: \n" 
    print semistandard_dist(dataBench)
    print "\n Ratio analysis: \n" 
    print ratios(data, benchSeries)
    print "\n One day VaR at the %.2f conf. level is: \n" % c
    print dataBench.apply(quant_var, args= (c,))
    print "\n One day loss beyond VaR (CVaR) at the %.2f conf. level is: \n" % c
    print dataBench.apply(quant_cvar, args= (c,))
    print "\n Correlation against benchmark: \n" 
    print correlations(data, benchSeries)
    print "\n Regression against benchmark: \n" 
    print regression(data, benchSeries)
    
    data1 = data.ix[:,0]
    print "\n Top ten drawdowns from: " + data1.name + "\n"
    dd_index = dd.drawdowns(data1)
    drawdowns = dd.all_dd(dd_index)
    print dd.max_dd(drawdowns)
    dd.dd_plot(drawdowns)
    dd.max_dd(drawdowns, n=10).to_csv('test_dd.csv')

    
if __name__ == '__main__':
    main()









  

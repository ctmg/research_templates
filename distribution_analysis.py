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

"""DATA"""


#'''Equity model study as an example'''
#equities = pd.read_csv('testdata_GemConsensusVsFMEquityIndices.csv', index_col=0, parse_dates=True)

'''CTA Index for correlation and beta - automate getting the data - is it in the WAB?
        http://www.newedge.com/content/newedgecom/en/home.html
            *the file requires addition of header, changing from percentage to number in excel'''
necta = pd.read_csv('Newedge_CTA_Index_historical_data.csv', index_col=0, header=None, parse_dates=True)['2004':]
necta.columns = ['ROR', 'M', 'A']; necta.index.name = 'Date'
necta.ix[:2] = 0

'''PQA data for testing
pqa = pd.read_csv('PQARoR_080114.csv', index_col=0, parse_dates=True)
pqa_m = pqa.resample('BM', how='sum')
pqa_q = pqa.resample('BQ-DEC', how='sum')
pqa_y = pqa.resample('BA-DEC', how='sum')
'''




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


def ratios(df, bench=necta['ROR']):
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



def regression(df, bench=necta['ROR']):
    return df.apply(lambda x: (pd.ols(y=x, x=bench).summary_as_matrix.x))
    
    

def correlations(df, bench=necta['ROR']):
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





def main(ror_file='PQARoR_080114.csv', c=.99):
        
    data = pd.read_csv(ror_file, index_col=0, parse_dates=True)
    #necta = pd.read_csv('Newedge_CTA_Index_historical_data.csv', index_col=0, header=1, parse_dates=True)['2004':]
    ret_index = data.apply(return_index)
    
    distribution = moments(data)
    print "\n The distribution looks like: \n"    
    print (distribution)
    print "\n The downside risk from all neg returns: \n" 
    print semistandard_dist(data)
    print "\n Ratio analysis: \n" 
    print ratios(data)
    print "\n One day VaR at the %.2f conf. level is: \n" % c
    print data.apply(quant_var, args= (c,))
    print "\n One day loss beyond VaR (CVaR) at the %.2f conf. level is: \n" % c
    print data.apply(quant_cvar, args= (c,))
    print "\n Correlation against NECTA: \n" 
    print correlations(data)
    print "\n Regression against NECTA: \n" 
    print regression(data)
    print "\n Top ten drawdowns: \n"
    data1 = data.ix[:,0]
    dd_index = dd.drawdowns(data1)
    drawdowns = dd.all_dd(dd_index)
    print dd.max_dd(drawdowns)
    dd.dd_plot(drawdowns)
    dd.max_dd(dd, n=10).to_csv('test_dd.csv')
    
    
if __name__ == '__main__':
    main()









  

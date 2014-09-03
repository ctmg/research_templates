# -*- coding: utf-8 -*-
"""

Drawdown algo

@author: colin4567
"""


import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pylab as plt
import pandas.io.data as web
import math
pd.set_option('notebook_repr_html', False)


"""FUNCTIONS"""

 
def return_index(ret, rorStyle):
    #use with apply
    ret.ix[0] = 1
    if rorStyle == 0:
        vami = ret.cumsum()
    elif rorStyle == 1:
        vami = ret.cumsum().apply(np.exp)
    return vami 

    
def drawdowns(ror, rorStyle=0):
    #use with apply
    vami = return_index(ror, rorStyle)
    print vami.tail()
    peak = pd.expanding_max(vami)
    t = pd.concat([ror, vami, peak], axis=1); t.columns = ['ror','vami','peak']
    t['indd'] = t.apply(lambda x: 1 if (x['peak'] > x['vami']) else np.nan, axis=1)
    #this is for compounding - needed?
    #t['indd'] = t.apply(lambda x: ((x['vami'] / x['peak'])-1) if (x['peak'] > x['vami']) else np.nan, axis=1) 
    return t


def all_dd(df):
    
    df.reset_index(level=0, inplace=True)
    df['start'] = pd.NaT; df['end'] = pd.NaT;  df['valley'] = pd.NaT; df['length_of_dd'] = np.nan; df['dd'] = np.nan; df['max_dd'] = np.nan

    for x in df.index[2:]:
        test = df.ix[x-1:x, 'indd']
        
        if (test.notnull()[x] and test.isnull()[x-1]):
            df.ix[x,'start'] = df.ix[x, 'Date']
            
        if (test.notnull()[x-1] and test.isnull()[x]): 
            s = df.ix[:x,'start'].last_valid_index()  
            df.ix[s:x-1,'end'] = df.ix[x-1, 'Date']
            #find valley 
            v = df.ix[s:x-1,'vami'].idxmin()
            df.ix[s:x-1, 'valley'] = df.ix[v,'Date']            
            df.ix[x-1,'length_of_dd'] = len(df[s:x-1])+1
            #need to sum ror's here
            df.ix[s:x-1, 'dd'] = pd.expanding_sum(df.ix[s:x-1,'ror'])
            df.ix[x-1, 'max_dd'] = df.ix[s:v,'ror'].sum()
        
        if (x == len(df)-1 and test.notnull()[x]):
            s = df.ix[:x,'start'].last_valid_index()
            df.ix[s:x,'end'] = df.ix[x, 'Date']
            #find valley 
            v = df.ix[s:x,'vami'].idxmin()
            df.ix[s:x, 'valley'] = df.ix[v,'Date']  
            df.ix[x,'length_of_dd'] = len(df[s:x])+1
            #need to sum ror's here
            df.ix[s:x, 'dd'] = pd.expanding_sum(df.ix[s:x,'ror'])
            df.ix[x, 'max_dd'] = df.ix[s:v,'ror'].sum()
             

    #forward fill the start 
    df['start'] = df['start'][df['indd'].notnull()].fillna(method='ffill')
    return df




def max_dd(draws, n=10, column='max_dd'):
    clean_draws = draws.dropna()
    top = clean_draws.sort_index(by=column)[:n]
    return top.set_index('max_dd')[['length_of_dd', 'start', 'valley', 'end']]




def dd_plot(df):
    df.set_index(['Date'], inplace=True)
    dd = df['dd'].fillna(0)
    fig, axes = plt.subplots(2,1)
    df['vami'].plot(ax=axes[0], title='Return Index')
    dd.plot(ylim=(dd.min()-.1, 0), ax=axes[1], title='Drawdown')



def main(ror_file='PQARoR_080114.csv', c=.99):
        
     data = pd.read_csv(ror_file, index_col=0, parse_dates=True)
     data1 = data.ix[:,0]
     dd_index = drawdowns(data1, rorStyle)
     dd = all_dd(dd_index)
     print max_dd(dd)
     return dd_plot(dd)
     #max_dd(dd, n=10).to_csv('test_dd.csv')
    
    

if __name__ == '__main__':
    main()









  

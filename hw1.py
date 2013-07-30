'''
Created on 2013-3-11

Start Date: January 1, 2011
End Date: December 31, 2011
Symbols: ['AAPL', 'GLD', 'GOOG', 'XOM']
Optimal Allocations: [0.4, 0.4, 0.0, 0.2]
Sharpe Ratio: 1.02828403099
Volatility (stdev of daily returns):  0.0101467067654
Average Daily Return:  0.000657261102001
Cumulative Return:  1.16487261965
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize


print "Pandas Version", pd.__version__

def simulate(dt_start, dt_end, ls_symbols, ls_allocations):
  
  # get closing prices at 16
  dt_timeofday = dt.timedelta(hours=16)
  
  # Get a list of trading days between the start and end
  ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
  
  # an object to get data from yahoo!
  c_dataobj = da.DataAccess('Yahoo')
  
  # Keys to read from the data
  ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
  
  # Read in adjusted closing prices for the 4 equities.
  # d_data is a dictionary with the keys value
  ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
  d_data = dict(zip(ls_keys, ldf_data))
  
  # Getting numpy ndarray of close prices
  na_price = d_data['close'].values
  
  # Normalize the price
  na_normalized_price = na_price / na_price[0, :]
  
  # Copy the normalized prices to a new ndarray to find returns
  na_rets = na_normalized_price.copy()
  #print "na_rets: \n", na_rets, "\n"
  
  #Calculate the daily returns of the prices. (Inplace calculation)
  tsu.returnize0(na_rets)
  
  #print "na_rets:\n", na_rets, "\n"
  #print "weighted na_rets:\n", np.dot(na_rets,np.transpose(ls_allocations)),"\n"
  #print "weight:\n", np.std(np.dot(na_rets,np.transpose(ls_allocations)))
  
  # Get weighted rets
  na_weighted_rets = np.dot(na_rets,np.transpose(ls_allocations))
  vol = np.std(na_weighted_rets)
  #print "Volatility: ", vol
  
  # Get Average Daily Return
  daily_ret = np.average(na_weighted_rets)
  #print "Average Daily Return: ", daily_ret
  
  # Get Sharpe Ratio
  k = 252**0.5
  sharpe = k * daily_ret / vol
  #print "Sharpe Ratio: " , sharpe
  
  # Get Cumulate Return
  cum_ret = np.dot(na_price[-1]/na_price[0], np.transpose(ls_allocations))
  #print "na_rets: \n", na_rets
  #print "na_rets -1 : \n", na_rets[-1]
  #print "Cumulative Return: ", cum_ret
  
  return vol, daily_ret, sharpe, cum_ret

def AntiSharpeRatio(ls_allocations, dt_start, dt_end, ls_symbols ):
  # get closing prices at 16
  dt_timeofday = dt.timedelta(hours=16)
  
  # Get a list of trading days between the start and end
  ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
  
  # an object to get data from yahoo!
  c_dataobj = da.DataAccess('Yahoo')
  
  # Keys to read from the data
  ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
  
  # Reading the data
  # d_data is a dictionary with the keys value
  ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
  d_data = dict(zip(ls_keys, ldf_data))
  
  # Getting numpy ndarray of close prices
  na_price = d_data['close'].values
  
  # Normalize the price
  na_normalized_price = na_price / na_price[0, :]
  
  # Copy the normalized prices to a new ndarray to find returns
  na_rets = na_normalized_price.copy()
  #print "na_rets: \n", na_rets, "\n"
  
  #Calculate the daily returns of the prices. (Inplace calculation)
  tsu.returnize0(na_rets)
  
  #print "na_rets:\n", na_rets, "\n"
  #print "weighted na_rets:\n", np.dot(na_rets,np.transpose(ls_allocations)),"\n"
  #print "weight:\n", np.std(np.dot(na_rets,np.transpose(ls_allocations)))
  
  # Get weighted rets
  na_weighted_rets = np.dot(na_rets,np.transpose(ls_allocations))
  vol = np.std(na_weighted_rets)
  #print "Volatility: ", vol
  
  # Get Average Daily Return
  daily_ret = np.average(na_weighted_rets)
  #print "Average Daily Return: ", daily_ret
  
  # Get Sharpe Ratio
  k = 252**0.5
  sharpe = k * daily_ret / vol
  
  return -sharpe

def constrain(x):
  return np.sum(x)-1

def OptimizeByScipy(dt_start, dt_end, ls_symbols):
  init_allocation = [0.25, 0.25, 0.25, 0.25]
  cons = ({'type': 'eq', 'fun': constrain})
  arg = (dt_start, dt_end, ls_symbols)
  #arg = (dt_start, dt_end, ls_symbols)
  bnds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0,1.0))
  res = minimize(fun=AntiSharpeRatio,x0=init_allocation,
                 args=arg,
                 method='SLSQP', 
                 constraints=cons,
                 bounds=bnds) 
  print res


def optimize(dt_start, dt_end, ls_symbols):
  optimal_sharpe = -100
  optimal_allocation = [0.0, 0.0, 0.0, 0.0]
  optimal_vol = 0
  optimal_daily_ret = 0
  optimal_cum_ret = 0
  count = 0
  for x in np.arange(0, 11):
    for y in np.arange(0, 11-x):
      for z in np.arange(0, 11-x-y):
        count = count + 1
        ls_allocations = [x/10.0, y/10.0, z/10.0, (10-x-y-z)/10.0]
        #print 'ls_allocations: ',ls_allocations
        vol, daily_ret, sharpe, cum_ret = simulate(dt_start, dt_end, ls_symbols, ls_allocations)
        if sharpe > optimal_sharpe:
          optimal_sharpe = sharpe
          optimal_vol = vol
          optimal_daily_ret = daily_ret
          #optimal_sharpe = sharpe
          optimal_cum_ret = cum_ret
          optimal_allocation = ls_allocations
  
  print count
  print " Optimal sharpe: " + str(optimal_sharpe)
  print " Optimal vol: " + str(optimal_vol)
  print " Optimal daily return: " + str(optimal_daily_ret)
  print " Optimal cumulative return:  " + str(optimal_cum_ret)
  print " Optimal allocation: " + str(optimal_allocation)
  
  return optimal_vol, optimal_daily_ret, optimal_sharpe, optimal_cum_ret, optimal_allocation
  
def test(function):
  precision = 8

  # Example inputs
  start = dt.datetime(2011, 1, 1)
  end = dt.datetime(2011, 12, 31)
  symbols = ['AAPL', 'GOOG', 'XOM', 'GLD']
  allocations = [0.4, 0.0, 0.2, 0.4]

  # Example outputs to test against
  test_vol = round(0.0101892118984, precision)
  test_avg = round(0.000713781244655, precision)
  test_sharpe = round(1.11205126521, precision)
  test_cum_ret = round(1.16487261965, precision)

  # the simulation function should return a tuple with four values
  vol, avg, sharpe, cum_ret = function(start, end, symbols, allocations)

  if round(vol, precision) == test_vol:
    print "Volatility calculation works."
    print "  Value: " + str(vol)
  else:
    print "Volatility calculation does not work."
    print "  Value: " + str(vol)
    print "  Expected: " + str(test_vol)

  if round(avg, precision) == test_avg:
    print "Averaging works."
    print "  Value: " + str(avg)
  else:
    print "Averaging does not work."
    print "  Value: " + str(avg)
    print "  Expected: " + str(test_avg)

  if round(sharpe, precision) == test_sharpe:
    print "Sharpe ratio calculation works."
    print "  Value: " + str(sharpe)
  else:
    print "Sharpe ratio calculation does not work."
    print "  Value: " + str(sharpe)
    print "  Expected: " + str(test_sharpe)

  if round(cum_ret, precision) == test_cum_ret:
    print "Cumulative return calculation works."
    print "  Value: " + str(cum_ret)
  else:
    print "Cumulative return calculation does not work."
    print "  Value: " + str(cum_ret)
    print "  Expected: " + str(test_cum_ret)

if __name__ == '__main__':
  
  # start and end time
  dt_start = dt.datetime(2011,1,1)
  dt_end = dt.datetime(2011,12,31)
  

  # List of symbols
  ls_symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
  ls_symbols = ['AAPL', 'GOOG', 'IBM', 'MSFT']
  # List of allocations
  ls_allocations = [0.4, 0.0, 0.2, 0.4]
  
  print optimize(dt_start, dt_end, ls_symbols)
  print simulate(dt_start, dt_end, ls_symbols, ls_allocations)
  test(simulate)
  res = OptimizeByScipy(dt_start, dt_end, ls_symbols)
  print res
  
  
  
  
  
  
  
  
  
  
  
  
  

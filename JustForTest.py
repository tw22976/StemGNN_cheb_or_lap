import pandas_datareader as pdr

stock_list = pdr.av.time_series.AVQuotesReader().symbols
print(stock_list)

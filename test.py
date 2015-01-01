############################################
#    Using Pandas for Stock Analysis
#    Author: Allen Long Chen
#    Date: 12/31/2014
############################################

import numpy as np
import datetime as dt
import pandas as pd
import pandas.io.data
import pandas.stats.moments as pa
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.finance import candlestick

# Load time series data from Yahoo finance
aapl = pd.io.data.get_data_yahoo('AAPL', # OK to use list for multiple tickers 
                                 start=dt.datetime(2014, 7, 31),
                                 end=dt.datetime(2014, 10, 31))

# Select close price as plotting data
close_px = aapl['Close']

# Parameters for Bollinger Bands
period = 10
std = 2

# Calculation of Bollinger Bands: SMA, Upper and Lower
mavg = pa.rolling_mean(close_px, period)
mstd = pa.rolling_std(close_px, period)
uband = mavg + 2 * mstd
lband = mavg - 2 * mstd

# Excercise: Use Matplotlib to plot stock price
#close_px.plot(label='AAPL', style='k*')
#mavg.plot(label='mavg')
#uband.plot()
#lband.plot()
#plt.legend()
#plt.show()

# Drop the date index from the dateframe and store as a time series date
aapl.reset_index(inplace = True)
ts_date = aapl.Date

# Use numeric sequence to skip weekend
N = len(ts_date)
ind = np.arange(N)  # the evenly spaced plot indices

# Convert Dates to ordinal integers for use of candlestick plot
aapl.Date = aapl.Date.map(dt.datetime.toordinal)
aapl.Date = ind
# Create list as input for candlestick plot
dataAr = [tuple(x) for x in aapl[['Date','Open','Close','High','Low']].to_records(index=False)]

# Tick locator and formatter
mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
#dayFormatter = DateFormatter('%Y-%m-%d')      # e.g., 12
dayFormatter = DateFormatter('%b %d %Y')

# Plot
fig = plt.figure()
ax = plt.subplot(1,1,1)
fig.autofmt_xdate()

# Write a custom formatter to convert integer index back to date

def format_date(x, pos=None):
    thisind = np.clip(int(x+0.5), 0, N-1)
    return ts_date[thisind].strftime('%b %d %Y')

ax.plot(ind, uband, '#2E2EFE', label='Bollinger Band')
ax.plot(ind, lband, '#2E2EFE')
ax.plot(ind, mavg, '#FE9A2E', label='SMA')
candlestick(ax, dataAr)

# Optional for traditional Date formatted axis
#ax.xaxis.set_major_locator(alldays)
#ax.xaxis.set_minor_locator(alldays)
#ax.xaxis.set_major_formatter(dayFormatter)

# Limit x-axis to the range of integer indices only
plt.xlim([0, N-1])

# Convert back to date
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
fig.autofmt_xdate()

plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('AAPL')
ax.autoscale_view()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='right')

# Hide the right and top spines/axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.tight_layout()
plt.legend(prop={'size':10})
plt.show()


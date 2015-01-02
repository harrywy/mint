############################################
#    Using Pandas for Stock Analysis
#    Author: Allen Long Chen
#    Date: 01/01/2015
############################################

import numpy as np
import datetime as dt
import pandas as pd
import pandas.io.data
import pandas.stats.moments as pa
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.finance import candlestick

import talib as tl


# Load time series data from Yahoo finance
aapl = pd.io.data.get_data_yahoo('AAPL', # OK to use list for multiple tickers 
                                 start=dt.datetime(2014, 8, 31),
                                 end=dt.datetime(2014, 12, 31))

# Use numeric sequence to skip weekend
N = len(aapl.Close)
ind = np.arange(N)  # the evenly spaced plot indices

# Write a custom formatter to convert integer index back to date
def format_date(x, pos=None):
    thisind = np.clip(int(x+0.5), 0, N-1)
    return ts_date[thisind].strftime('%Y-%m-%d')

####################################################
#     Bollinger Bands with Pandas                  #
####################################################

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

##################################################
#       Data for Candlestick Chart               #
#       Open, Close, High, Low                   #
##################################################

# Drop the date index from the dateframe and store as a time series date
aapl.reset_index(inplace = True)
ts_date = aapl.Date

# Convert Dates to ordinal integers for use of candlestick plot
aapl.Date = aapl.Date.map(dt.datetime.toordinal)
aapl.Date = ind

# Create list as input for candlestick plot
dataAr = [tuple(x) for x in aapl[['Date','Open','Close','High','Low']].to_records(index=False)]


###########################################
#   Stochastics with TA-Lib               #
###########################################

high_price = aapl.High.values
low_price = aapl.Low.values
close_price = aapl.Close.values

# Parameters
fastk_period = 5
slowk_period = 3
slowk_matype = 0
slowd_period = 3
slowd_matype = 0

slowk, slowd = tl.STOCH(high_price, low_price, close_price,
                        fastk_period, slowk_period, slowk_matype,
                        slowd_period, slowd_matype
                       )

###########################################
#      Setup of Plotting                  #
###########################################

# Tick locator and formatter
mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%Y-%m-%d')      # e.g., 12
#dayFormatter = DateFormatter('%b %d %Y')

# Plot
fig = plt.figure(facecolor='#07000d')
ax1 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4, axisbg='#07000d')
ax1.grid(True, color='w')
#fig.autofmt_xdate()
ax1.plot(ind, uband, '#2E2EFE', label='Bollinger Band')
ax1.plot(ind, lband, '#2E2EFE')
ax1.plot(ind, mavg, '#FE9A2E', label='SMA')
candlestick(ax1, dataAr, width=0.8, colorup='#9eff15', colordown='#ff1717')
# Convert back to date
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
# Axis and tick setting
ax1.yaxis.label.set_color('w')
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.spines['bottom'].set_color("#5998ff")
ax1.spines['top'].set_color("#5998ff")
ax1.spines['left'].set_color("#5998ff")
ax1.spines['right'].set_color("#5998ff")
ax1.tick_params(axis='y', colors='w')
# Legend
leg1 = ax1.legend(loc=2, prop={'size':8})
for text in leg1.get_texts():
    text.set_color('w')
frame1 = leg1.get_frame()
frame1.set_facecolor('#07000d')
frame1.set_edgecolor('#07000d')
plt.ylabel('Stock price')
# Hide the right and top spines/axes
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
#ax1.xaxis.set_ticks_position('bottom')
#ax1.yaxis.set_ticks_position('left')

ax2 = plt.subplot2grid((4,4), (3,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
ax2.grid(False)
ax2.yaxis.label.set_color('w')
ax2.axhline(0, color='w', linestyle='-', linewidth=0.2)
ax2.axhline(20, color='w', linestyle='-', linewidth=0.2)
ax2.axhline(80, color='w', linestyle='-', linewidth=0.2)
ax2.axhline(100, color='w', linestyle='-', linewidth=0.2)
ax2.axes.yaxis.set_ticks([20,80], minor=False)
ax2.spines['bottom'].set_color("#5998ff")
ax2.spines['top'].set_color("#5998ff")
ax2.spines['left'].set_color("#5998ff")
ax2.spines['right'].set_color("#5998ff")
ax2.tick_params(axis='x', colors='w')
ax2.tick_params(axis='y', colors='w')
ax2.plot(ind, slowk, label="%K")
ax2.plot(ind, slowd, label="%D")
# Limit Date to scope
ax2.set_xlim([0, N-1])
# Set ticks
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('none')
# Legend
leg2 = ax2.legend(loc=2, prop={'size':8})
for text in leg2.get_texts():
    text.set_color('w')
frame2 = leg2.get_frame()
frame2.set_facecolor('#07000d')
frame2.set_edgecolor('#07000d')
plt.ylabel('Stochastics')

# Optional for traditional Date formatted axis
#ax2.xaxis.set_major_formatter(dayFormatter)
#ax2.xaxis.set_major_locator(mondays)
#ax2.xaxis.set_minor_locator(alldays)
#ax2.xaxis.set_major_formatter(dayFormatter)

plt.xlabel('Date', color='w')
plt.suptitle('AAPL', color='w')
#plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='right')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='right')
#plt.tight_layout()
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(left=.10, bottom=.18, right=.95, top=.94, wspace=.20, hspace=0)
plt.show()


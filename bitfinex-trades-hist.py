#!/usr/bin/env python
#
# Poll the Bitfinex order book and print to console.

import os, sys
import multiprocessing, math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time 
import csv, json

from multiprocessing import Process, Queue, Manager

from client import Test, Helper

# create the client
API_KEY=os.getenv("BFX_KEY")
API_SECRET=os.getenv("BFX_SECRET")

test = Test(API_KEY, API_SECRET)
helper = Helper(API_KEY, API_SECRET)
os.system('clear')

symbol = 'BTCUSD'
dates, open_data, close_data, high_data, low_data, volume_data = helper.getCandles("3h",symbol)

fig = make_subplots(rows=2, cols=1)
fig_quick = go.Figure()

# create cutout
start = len(close_data)-9900#3300 #3500 #334 #2500 #276 #830 #360 #87 #9950 #2920 #8760
end = len(close_data)-0


# datesShortCut = dates_short[start_short:]
# datesCut = dates[start:end]
# close_dataCut = close_data[start:end]
# low_dataCut = low_data[start:end]
# high_dataCut = high_data[start:end]
# open_dataCut = open_data[start:end]

# candles
# candle_fig = go.Candlestick(x=datesCut,
#                 open=open_dataCut, high=high_dataCut,
#                 low=low_dataCut, close=close_dataCut)
# fig.append_trace(candle_fig, row=1, col=1)
# fig.update_layout(xaxis_rangeslider_visible=False)

#heikin-ashi
ash_c, ash_o, ash_h, ash_l, ash_d, ash_vol = [], [], [], [], [], []
ash_d = dates[1:]
ash_vol = volume_data[1:]

for x in range(1, len(close_data)):
    ash_c.append(0.25 * (close_data[x] + open_data[x] + close_data[x] + low_data[x]))
    ash_o.append(0.5 * (open_data[x-1]+ close_data[x-1]))
    ash_h.append(max(high_data[x], open_data[x], close_data[x]))
    ash_l.append(min(low_data[x], open_data[x], close_data[x]))

# ash_candle_fig = go.Candlestick(x=datesCut,
#                 open=ash_o_cut, high=ash_h_cut,
#                 low=ash_l_cut, close=ash_c_cut)
# fig.append_trace(ash_candle_fig, row=3, col=1)
# fig.update_layout(xaxis_rangeslider_visible=False)

""" # rsi
rsi = test.rsi(10, ash_c)
stoch_rsi = test.stoch_rsi(10, rsi, 3, 3)
balance_rsi = test.rsi_execute(10, stoch_rsi, ash_c, 95, 5, dates, start)
fig_quick.add_trace(go.Scatter(y=balance_rsi[0], x=balance_rsi[1], name='rsi-ash'))

# stoch_rsi = stoch_rsi[start:end]
# fig_quick.add_trace(go.Scatter(y=stoch_rsi, x=datesCut, name='rsi-ash'))

rsi = test.rsi(10, close_data)
stoch_rsi = test.stoch_rsi(10, rsi, 3, 3)
balance_rsi = test.rsi_execute(10, stoch_rsi, close_data, 95, 5, dates, start)
fig_quick.add_trace(go.Scatter(y=balance_rsi[0], x=balance_rsi[1], name='rsi'))
# stoch_rsi = stoch_rsi[start:end]
# fig_quick.add_trace(go.Scatter(y=stoch_rsi, x=datesCut, name='rsi')) """


""" # rsi
rsi = test.rsi(10, close_data)
stoch_rsi = test.stoch_rsi(10, rsi, 3, 3)

balance_rsi = test.rsi_execute(10, stoch_rsi, close_data, 90, 10, dates, start)
fig_quick.add_trace(go.Scatter(y=balance_rsi[0], x=balance_rsi[1], name='rsi'))

#moving average
averageShort = test.moving_average(close_data, 14) 
averageLong = test.moving_average(close_data, 25)
cross = test.ma_cross(averageShort, averageLong)

balance = test.mAV_execute(cross, close_data, dates, start)
fig_quick.add_trace(go.Scatter(y=balance[0], x=balance[1], name='mAV:knowing'))
#fig_quick.add_trace(go.Scatter(y=balance[0], x=balance[1], name='mAV:knowing'))

#mfi
mfi = test.mfi_execute(close_data, dates, high_data, low_data, volume_data, start, 80, 20)

fig_quick.add_trace(go.Scatter(y=mfi[0], x=mfi[1], name='mfi:balance'))

#combined
rsi = test.rsi(14, close_data)
stoch_rsi = test.stoch_rsi(14, rsi, 3, 3)
# --> this is a method where only buy is possible without limitation of amount != 0
rsi_calculation = test.rsi_execute(8, stoch_rsi, close_data, 100, 5, dates, start)[2]

averageShort = test.moving_average(close_data, 7)
averageLong = test.moving_average(close_data, 22)
mAV_calculation = test.ma_cross(averageShort, averageLong, dates)

balance = test.combined(close_data, dates, rsi_calculation, mAV_calculation, start, len(close_data), low_data, high_data, mav)

fig_quick.add_trace(go.Scatter(y=balance[0], x=balance[1], name='comb:balance:knowing'))

# fig_quick.show() """


# --------------------------      ash rsi + mav + mfi      --------------------------------------#
####
low = 5
i, mfiP = 17, 17
mavLong, mavShort = 23, 9
mfiLow, mfiHigh = 15, 90
limitStop = 0

#rsi
rsi = test.rsi(i, ash_o)
stoch_rsi = test.stoch_rsi(i, rsi, 3, 1)
rsi_calculation = test.rsi_execute(i, stoch_rsi, ash_o, 100, low, ash_d, start, limitStop)[2]
rsi_calculation[0] = rsi_calculation[0][start+1:]
rsi_calculation[1] = rsi_calculation[1][start+1:]
rsi_calculation[2] = rsi_calculation[2][start+1:]

# stoch_rsi = stoch_rsi[start:]
# fig.append_trace(go.Scatter(y=stoch_rsi, x=datesCut, name='rsi-long'), row=2, col=1)

#mav
averageShort = test.moving_average(ash_c, mavShort)
averageLong = test.moving_average(ash_c, mavLong)
mAV_calculation = test.ma_cross(averageShort, averageLong, ash_d)
mAV_calculation[0] = mAV_calculation[0][start+1:]
mAV_calculation[1] = mAV_calculation[1][start+1:]

#mfi
mfi_calculation = test.mfi_execute(ash_c,ash_d,ash_h, ash_l, ash_vol, start, mfiHigh, mfiLow, mfiP)[2]
mfi_calculation[0] = mfi_calculation[0][start+1:]
mfi_calculation[1] = mfi_calculation[1][start+1:]
mfi_calculation[2] = mfi_calculation[2][start+1:]
#combMFI
balance = test.combinedMFI(close_data[1:], dates[1:], rsi_calculation, mAV_calculation, start, end, low_data[1:], high_data[1:], mfi_calculation, open_data[1:])
# for i, bal in enumerate(balance[0]):
#     balance[0][i] = math.log(bal)
f = open("maxBalanceAsh.txt", "w")
f.write(str(balance[0][len(balance[0])-1]))
f.close()
fig_quick.add_trace(go.Scatter(y=balance[0], x=balance[1], name='ash'+str([low, i, mavShort, mavLong, mfiHigh])+ 'close stoploss45'))

# --------------------------      normal rsi + mav + mfi      --------------------------------------#
#####
low = 5
i, mfiP = 14, 33
# mavLong, mavShort = 64, 12
mavLong, mavShort = 86, 4
mfiLow, mfiHigh = 15, 85
limitStop = 0

#rsi
rsi = test.rsi(i, open_data)
stoch_rsi = test.stoch_rsi(i, rsi, 3, 3)
rsi_calculation = test.rsi_execute(i, stoch_rsi, open_data, 100, low, dates, start, limitStop)[2]

rsi_calculation[0] = rsi_calculation[0][start+1:]
rsi_calculation[1] = rsi_calculation[1][start+1:]
rsi_calculation[2] = rsi_calculation[2][start+1:]

#mav
averageShort = test.moving_average(close_data, mavShort)
averageLong = test.moving_average(close_data , mavLong)
print(averageLong[len(averageLong)-1], averageShort[len(averageShort)-1])
mAV_calculation = test.ma_cross(averageShort, averageLong, dates)

mAV_calculation[0] = mAV_calculation[0][start+1:]
mAV_calculation[1] = mAV_calculation[1][start+1:]

#mfi
mfi_calculation = test.mfi_execute(close_data, dates, high_data, low_data, volume_data, start, mfiHigh, mfiLow, mfiP)[2]

mfi_calculation[0] = mfi_calculation[0][start+1:]
mfi_calculation[1] = mfi_calculation[1][start+1:]
mfi_calculation[2] = mfi_calculation[2][start+1:]
# mfi_calculation[2] = test.scale(mfi_calculation[2])

#combMFI
balance = test.combinedMFI(close_data, dates, rsi_calculation, mAV_calculation, start, len(close_data), low_data, high_data, mfi_calculation, open_data)
f = open("maxBalanceNormal.txt", "w")
f.write(str(balance[0][len(balance[0])-1]))
f.close()
# for i, bal in enumerate(balance[0]):
#     balance[0][i] = math.log(bal)
fig_quick.add_trace(go.Scatter(y=balance[0], x=balance[1], name='normal'+str([low, i, mavShort, mavLong, mfiHigh])+ 'close stoploss45'))
fig_quick.show()
# fig.show()

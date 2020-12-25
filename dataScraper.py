import os, sys
from datetime import datetime
import numpy as np
import time
from client import Helper

API_KEY=os.getenv("BFX_KEY")
API_SECRET=os.getenv("BFX_SECRET")
helper = Helper(API_KEY, API_SECRET)
os.system('clear')

PATH_CANDLES = "candles/trade%s"
VERSION2 = "v2"
SYMBOL = 'BTCUSD'
timeFrame = '30m'
parameters = {'limit': 10000, 'sort': -1}
timeFrame_symbol = ":" + timeFrame + ":t" + SYMBOL

# dates, open_data, close_data, high_data, low_data, volume_data = helper.getCandles("30m",symbol)
candles = helper._get(helper.url_for(PATH_CANDLES+'/hist', (timeFrame_symbol), parameters=parameters, version=VERSION2))

while datetime.utcfromtimestamp(int(candles[len(candles)-1][0]) / 1000).strftime('%Y-%m-%d %H:%M:%S') >= '2017-09-01 00:00:00':
    parameters = {'limit': 10000, 'sort': -1, 'end': candles[len(candles)-1][0]}
    data = helper._get(helper.url_for(PATH_CANDLES+'/hist', (timeFrame_symbol), parameters=parameters, version=VERSION2))
    data.pop(0)

    candles.extend(data)

    time.sleep(2)
    print("current earliest date: ", datetime.utcfromtimestamp(int(candles[len(candles)-1][0]) / 1000).strftime('%Y-%m-%d %H:%M:%S'))

print("start converting to np array")
candles = np.array(candles)
# open a binary file in write mode
file = open("candles30m_2017:09:01", "wb")
# save array to the file
np.save(file, candles)
# close the file
file.close
print("finished")
sys.exit()
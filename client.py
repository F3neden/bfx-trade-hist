from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import requests
import json
import base64
import hmac
import hashlib
import time
from datetime import datetime
from tenacity import *

import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense


PROTOCOL = "https"
HOST = "api.bitfinex.com"
VERSION = "v1"
VERSION2 = "v2"

PATH_SYMBOLS = "symbols"
PATH_CANDLES = "candles/trade%s"
PATH_TRADES = "trades/%s/hist"

# HTTP request timeout in seconds
TIMEOUT = 5.0

class Test:
    def __init__(self, key, secret):
        self.helper = Helper(key, secret)
        pass
    
    def calc_balance(self, price, amount):
        self.balance = price * amount 
        return self.balance

    def getCandles3h(self, timeFrame_symbol, parameters, section):
        return self.helper._get_3h(self.helper.url_for(PATH_CANDLES+section, (timeFrame_symbol), parameters=parameters, version=VERSION2))

    def rsi_step_one(self, close_data, period):
        """
        1. calculate for the first 'period' days
            1.1 the average gain 
                if there is a gain, then sum it up and divide it by the number of days with a gain
            1.2 the average loss
                if there is a loss, then sum it up and divide it by the number of days with a loss
            1.3 add 1 to the quotient of the average gain divided by the average loss
            1.4 100 - 100 / '1.3'
        """
        av_gain = 0
        av_loss = 0
        for x in range(0, period):
            relative_change = close_data[x] - close_data[x+1]
            " if negative = gain "
            if relative_change < 0:
                av_gain += abs(relative_change)
            elif relative_change > 0:
                av_loss += abs(relative_change)      

        av_gain = av_gain / period
        av_loss = av_loss / period

        if av_loss == 0:
            rsi = 100
        elif av_gain == 0:
            rsi = 0
        else:
            rsi = 100 - ( 100 / (1 + av_gain/av_loss))
        
        return rsi, av_gain, av_loss

    def rsi_step_two(self, period, close_data, gain, loss, day):
        """
        2. calculate for every following day
            2.1 add the current gain/loss to the previous average gain/loss which is multiplied with period -1
        """
        relative_change = close_data[day-1] - close_data[day]
        " if negative = gain "
        current_gain, current_loss = 0, 0

        if relative_change < 0:
            current_gain = abs(relative_change)
        elif relative_change > 0:
            current_loss = abs(relative_change)   

        av_gain = (gain*(period-1) + current_gain) / period
        av_loss = (loss*(period-1) + current_loss) / period

        if av_loss == 0:
            rsi = 100
        elif av_gain == 0:
            rsi = 0
        else:
            rsi = 100 - ( 100 / (1 + av_gain / av_loss ))

        return rsi, av_gain, av_loss

    def rsi(self, period, close_data):
        rsi_data = []

        for x in range(0, period-1):
            rsi_data.append(None)

        step_one = self.rsi_step_one(close_data, period)
        rsi_data.append(step_one[0])
        av_gain = step_one[1]
        av_loss = step_one[2]

        # calculate for all the following days
        for y in range(period, len(close_data)):
            step_two = self.rsi_step_two(period, close_data, av_gain, av_loss, y)
            av_gain = step_two[1]
            av_loss = step_two[2]
            rsi_data.append(step_two[0])

        return rsi_data

    def stoch_rsi(self, period, rsi_data, smooth, smooth2):
        stoch_rsi_data = []
        lowest, highest, stoch_rsi = 0, 0, 0
        x = 0
        while(rsi_data[x] == None):
            x +=1

        for i in range(0, period+x-1):
            stoch_rsi_data.append(None)

        for y in range(x, len(rsi_data)-period+1):
            highest = -1
            lowest = 101
            for day in range(y, y+period):
                if rsi_data[day] > highest:
                    highest = rsi_data[day]
                if rsi_data[day] < lowest:
                    lowest = rsi_data[day]
            
            if (highest - lowest) == 0:
                stoch_rsi_data.append(None)
            else:
                stoch_rsi = 100*((rsi_data[day] - lowest) / (highest - lowest))
                stoch_rsi_data.append(stoch_rsi)
                # stoch_rsi_data.append(stoch_rsi/100)

        stoch_rsi_data = self.moving_average(stoch_rsi_data, smooth)
        stoch_rsi_data = self.moving_average(stoch_rsi_data, smooth2)

        return stoch_rsi_data
    
    def scale(self, inputArr):
        ret = []
        for element in inputArr:
            if not element == None:
                ret.append(element/100)
        return ret

    def mfi_execute(self, close_data, dates, high_data, low_data, volume_data, startAt, limit_top, limit_bottom, period):
        if period == None:
            period = 17
        mfi_data = self.calculate_MFI(high_data, low_data, close_data, volume_data, period, dates)
        balances, b_dates, buyOrSell = [], [], [[],[], []]
        buyOrSell[0].append(None)
        buyOrSell[1].append(0)
        buyOrSell[2].append(None)
        self.balance = 100

        if startAt < period*2:
            start = period
        else:
            start = startAt
            
        buy = close_data[start]
        amount = (self.balance - 0.002*self.balance)/buy
        """ print("")
        print("")
        print("")
        print("")
        print("starting balance -> mfi:knowing: ", self.calc_balance(buy, amount)) """
        self.calc_balance(buy, amount)
        balances.append(self.balance)
        b_dates.append(dates[start])
        """ print(dates[start], " [mfi:knowing] buy at ", close_data[start]) """

        for x in range(0, start):
            buyOrSell[1].append(dates[x])
            buyOrSell[0].append(None)
            buyOrSell[2].append(None)

        for day in range(start, len(mfi_data)):
            buyOrSell[1].append(dates[day])
            buyOrSell[2].append(mfi_data[day])
            #overbought = sell
            if mfi_data[day] < limit_top and mfi_data[day-1] > limit_top:
                buyOrSell[0].append("sell")

            #oversold = buy
            elif mfi_data[day] > limit_bottom and mfi_data[day-1] < limit_bottom:
                buyOrSell[0].append("buy")

            else:
                buyOrSell[0].append(None)


            #overbought = sell
            if ((mfi_data[day] < limit_top and mfi_data[day-1] > limit_top) or day == (len(mfi_data)-1)) and amount != 0:
                """ print(dates[day], " [mfi:knowing] sell at ", close_data[day], "       ", self.calc_balance(close_data[day], amount)) """
                self.calc_balance(close_data[day], amount)
                balances.append(self.balance)
                b_dates.append(dates[day])
                amount = 0

            #oversold = buy
            elif mfi_data[day] > limit_bottom and mfi_data[day-1] < limit_bottom and amount == 0:
                amount = (self.balance - 0.002*self.balance) / close_data[day]
                """ print(dates[day], " [mfi:knowing] buy at ", close_data[day])    """           

        return balances, b_dates, buyOrSell, mfi_data

    def rsi_execute(self, period, rsi_data, close_data, limit_top, limit_bottom, dates, startAt, limitStop):
        balances, b_dates, buyOrSell = [], [], [[],[], []]
        self.balance = 100
        lastBuyPrice = 0
        buyOrSell[0].append(None)
        buyOrSell[1].append(0)
        buyOrSell[2].append(0)

        if startAt < period*2+1:
            start = period*2+1
        else:
            start = startAt
            
        buy = close_data[start]
        amount = (self.balance - 0.002*self.balance)/buy
        """ print("")
        print("")
        print("")
        print("")
        print("starting balance -> stochRSI:knowing: ", self.calc_balance(buy, amount)) """
        self.calc_balance(buy, amount)
        balances.append(self.balance)
        b_dates.append(dates[start])
        """ print(dates[start], " [rsi:knowing] buy at ", close_data[start]) """

        for x in range(0, start):
            buyOrSell[0].append(None)
            buyOrSell[2].append(None)
            buyOrSell[1].append(dates[x])

        for day in range(start, len(rsi_data)):
            buyOrSell[1].append(dates[day])
            buyOrSell[2].append(rsi_data[day])

            #overbought = sell
            if rsi_data[day] <= limit_top and rsi_data[day-1] >= limit_top:
                buyOrSell[0].append("sell")

            #oversold = buy
            elif rsi_data[day] > limit_bottom-limitStop and rsi_data[day-1] < limit_bottom:
                buyOrSell[0].append("buy")

            else:
                buyOrSell[0].append(None)

            """ #overbought = sell
            if ((rsi_data[day] <= limit_top and rsi_data[day-1] >= limit_top) or day == (len(rsi_data)-1)) and amount != 0:
            #if rsi_data[day] > limit_bottom and rsi_data[day-1] < limit_bottom:
                print(dates[day], " [rsi:knowing] sell at ", close_data[day], "       ", self.calc_balance(close_data[day], amount))
                self.calc_balance(close_data[day], amount)
                balances.append(self.balance)
                b_dates.append(dates[day])
                amount = 0

            #oversold = buy
            elif rsi_data[day] >= limit_bottom and rsi_data[day-1] <= limit_bottom and amount == 0:
            #elif rsi_data[day] < limit_bottom and rsi_data[day-1] > limit_bottom and amount == 0:
                amount = (self.balance - 0.002*self.balance) / close_data[day]
                print(dates[day], " [rsi:knowing] buy at ", close_data[day]) """              

        return balances, b_dates, buyOrSell

    def moving_average(self, close_data, period):
        average = []
        x = 0

        while close_data[x] == None:
            x += 1
            average.append(None)

        for i in range(x+1, x+period):
            average.append(None)

        for y in range(x, len(close_data)-(period-1)):
            sum = 0
            periodDivisor = period
            for x in range(y, y+period):
                if close_data[x] == None:
                    periodDivisor -= 1
                else:
                    sum += close_data[x]

            if sum == 0 and periodDivisor == 0:
                average.append(0.0)
            else:    
                average.append(round(sum/periodDivisor, 4))

        return average

    def ma_cross(self, av1, av2, dates=None):
        x = 1
        # initialize and declare with empty beginning
        if dates == None:
            cross = []
            cross.append(None)
            cross.append(None)
        else:
            cross = [[],[]]
            cross[0].append(None)
            cross[1].append(0)
            cross[0].append(None)
            cross[1].append(dates[0])

        while av1[x-1] == None or av2[x-1] == None:
            if dates == None:
                cross.append(None)
            else:
                cross[0].append(None)
                cross[1].append(dates[x])
            x += 1

        for y in range(x, len(av1)):
            #if av1 crosses av2 from below to top  = buy
            if av1[y] > av2[y] and av1[y-1] < av2[y-1]:
                if dates == None:
                    cross.append("buy")
                else:
                    cross[0].append("buy")
                    cross[1].append(dates[y])
            #if av1 falls below av2 = sell
            elif av1[y] < av2[y] and av1[y-1] > av2[y-1]:
                if dates == None:
                    cross.append("sell")
                else:
                    cross[0].append("sell")
                    cross[1].append(dates[y])
            else:
                if dates == None:
                    cross.append(None)
                else:
                    cross[0].append(None)
                    cross[1].append(dates[y])
                    
        return cross

    def mAV_execute(self, cross, close_data, dates, startAt):
        balances, b_dates = [], []
        self.balance = 100
        amount = 0
        amount = (self.balance - 0.002*self.balance) / close_data[startAt]
        """ print("")
        print("")
        print("")
        print("")
        print("starting balance -> mAV:knowing: ", self.calc_balance(close_data[startAt], amount))
        print(dates[startAt], " [mAV:knowing] buy at ", close_data[startAt]) """
        self.calc_balance(close_data[startAt], amount)
        balances.append(self.balance)
        b_dates.append(dates[startAt])

        for day in range(startAt, len(cross)):
            #sell
            if ((cross[day] == "sell") or day == (len(cross)-1)) and amount != 0:
                #print(dates[day], " [mAV:knowing] sell at ", close_data[day], "       ", self.calc_balance(close_data[day], amount))
                self.calc_balance(close_data[day], amount)
                balances.append(self.balance)
                b_dates.append(dates[day])
                amount = 0
            
            #buy
            if cross[day] == "buy" and amount == 0:
                amount = (self.balance - 0.002*self.balance) / close_data[day]
                #print(dates[day], " [mAV:knowing] buy at ", close_data[day])

        return balances, b_dates

    def buy_or_sell(self, rsi_data, limit_top, limit_bottom, dates, start, cutBottom, end=None):
        buyOrSell = [[],[]]
        
        if end == None:
            for day in range(start, len(rsi_data)):
                buyOrSell[1].append(dates[day])
                #overbought = sell
                if rsi_data[day] < limit_top and rsi_data[day-1] > limit_top:
                    buyOrSell[0].append("sell")

                #oversold = buy
                elif rsi_data[day] > limit_bottom and rsi_data[day-1] < limit_bottom:
                    buyOrSell[0].append("buy")

                else:
                    buyOrSell[0].append(None)
        else:
            for day in range(start, end):
                buyOrSell[1].append(dates[day])
                #overbought = sell
                if rsi_data[day] < limit_top and rsi_data[day-1] > limit_top:
                    buyOrSell[0].append("sell")

                #oversold = buy
                elif rsi_data[day] > limit_bottom-cutBottom and rsi_data[day-1] < limit_bottom:
                    buyOrSell[0].append("buy")

                else:
                    buyOrSell[0].append(None)
        return buyOrSell
        """ #overbought = sell
        if rsi_data[len(rsi_data)-1] < limit_top and rsi_data[len(rsi_data)-2] > limit_top:
            return "sell", dates[len(dates)-1]

        #oversold = buy
        elif rsi_data[len(rsi_data)-1] > limit_bottom and rsi_data[len(rsi_data)-2] < limit_bottom:
            return "buy", dates[len(dates)-1]

        else:
            return None, dates[len(dates)-1] """

    def calculate_MFI(self, high_data, low_data, close_data, volume_data, period, dates=None):
        """
        MFI =  100 - ( 100 / (1+MFRatio))
        MFRatio = (Period PosMoneyFlow)/(Period NegMoneyFlow)
        RawMoneyFlow = TypicalPrice * Volume
        TypicalPrice = (High+Low+Close)/3
        """
        typicalPrice, typicalPriceBefore, MFRatio = [], 0, 0
        MFI = []

        for y in range(0, period-1):
            MFI.append(None)

        #1. calculate typical price
        for x in range(0, len(close_data)):
            typicalPrice.append(self.MFI_typicalPrice(close_data,high_data,low_data,x))

        for x in range(period-1, len(close_data)):
            PosMoneyFlow, NegMoneyFlow = 0, 0
            
            for day in range(x-period+1, x+1):
                higher = True

                try:
                    typicalPriceBefore = typicalPrice[day-1]
                except IndexError:
                    continue
                
                if typicalPriceBefore > typicalPrice[day]:
                    higher =  False

                #2. append raw Money Flow either positive or negative
                if higher:
                    PosMoneyFlow += (typicalPrice[day] * volume_data[day])
                else:
                    NegMoneyFlow += (typicalPrice[day] * volume_data[day])

            #3. calculate RAWMoneyFlow
            if NegMoneyFlow != 0:
                MFRatio = PosMoneyFlow/NegMoneyFlow
            else:
                MFRatio = PosMoneyFlow

            #4. MFI
            MFI.append(100- ( 100 / (1 + MFRatio )))

        return MFI

    def MFI_typicalPrice(self, close_data, high_data, low_data, day):
        return (high_data[day]+low_data[day]+close_data[day])/3

    def buy_or_sell_MFI(self, mfi, limitLow, limitHigh, dates):
        #overbought = sell
        if mfi[len(mfi)-1] < limitHigh and mfi[len(mfi)-2] > limitHigh:
            return "sell", dates[len(dates)-1]

        #oversold = buy
        elif mfi[len(mfi)-1] > limitLow and mfi[len(mfi)-2] < limitLow:
            return "buy", dates[len(dates)-1]

        else:
            return None, dates[len(dates)-1]

    def calcSupports(self, high, low, close, dates):
        """
        weekly data
        PP = (High + Low + Close) / 3
 		R1 = 2 * PP - Low
 		S1 = 2 * PP - High
 		R2 = PP + (High - Low)
 		S2 = PP - (High - Low)
 		R3 = PP + 2 * (High - Low)
		S3 = PP - 2 * (High - Low)
        """
        pp, s1, s2, s3, s4 = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]

        duration = 112

        for i in range(duration, len(high)-1):
            date = dates[i]
            maxV = -sys.maxsize
            minV = sys.maxsize
            for idx in range(i-duration-duration,i+1-duration):
                #max
                if maxV < close[idx]:
                    maxV = close[idx]
                #min
                if minV > close[idx]:
                    minV = close[idx]

            pp[0].append((maxV + minV + close[i]) / 3)
            pp[1].append(dates[i])

            curPP = pp[0][len(pp[0])-1]
            s1[0].append(2*curPP - maxV)
            s1[1].append(dates[i])

            s2[0].append(curPP - (maxV-minV))
            s2[1].append(dates[i])

            s3[0].append(curPP - 2*(maxV-minV))
            # s3[0].append(minV - 2*(maxV-curPP))
            s3[1].append(dates[i])

            s4[0].append(((close[i-1]/1000 -(close[i-1]/1000)%1))*1000)
            s4[1].append(dates[i])

        return s1, s2, s3, s4

    def calcResistances(self, high, low, close, dates):
        """
        weekly data
        PP = (High + Low + Close) / 3
 		R1 = 2 * PP - Low
 		S1 = 2 * PP - High
 		R2 = PP + (High - Low)
 		S2 = PP - (High - Low)
 		R3 = PP + 2 * (High - Low) ----- R3 = High + 2 *  (PP - Low)
		S3 = PP - 2 * (High - Low)
        """
        pp, r1, r2, r3, r4 = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]

        duration = 112

        for i in range(duration, len(high)-1):
            date = dates[i]
            maxV = -sys.maxsize
            minV = sys.maxsize
            for idx in range(i-duration-duration,i+1-duration):
                #max
                if maxV < close[idx]:
                    maxV = close[idx]
                #min
                if minV > close[idx]:
                    minV = close[idx]

            pp[0].append((maxV + minV + close[i]) / 3)
            pp[1].append(dates[i])

            curPP = pp[0][len(pp[0])-1]
            r1[0].append(int(round(2*curPP -minV)))
            r1[1].append(dates[i])

            r2[0].append(int(round(curPP + (maxV-minV))))
            r2[1].append(dates[i])

            r3[0].append(int(round(curPP + 2*(maxV-minV))))
            # r3[0].append(maxV + 2 * (curPP-minV))
            r3[1].append(dates[i])

            r4[0].append(((close[i-1]/1000 -(close[i-1]/1000)%1)+1)*1000)
            r4[1].append(dates[i])

        return r1, r2, r3, r4

    def call_simulation(self, close_data, dates, close_data_mAV, dates_mAV, start, start_mAV, low_data, high_data):
        rsi_calculation, mAV_calculation = [], []
        
        print("")
        print("")
        print("")
        print("")

        rsi_calculation = self.call_rsi(close_data, dates, start)
        #mAV_calculation = self.call_mAV(close_data_mAV, dates_mAV, start_mAV)
        mAV_calculation = self.call_mAV(close_data, dates, start)

        balance_combined = [[],[]] 
        balance_combined = self.combined(close_data, dates, rsi_calculation[3], mAV_calculation[2], start, low_data, high_data)

        #return rsi_balances,       rsi_b_dates,        mAV_balances,       mAV_b_dates,        rsi_data,           balance_combined
        return rsi_calculation[0], rsi_calculation[1], mAV_calculation[0], mAV_calculation[1], rsi_calculation[2], balance_combined
        #return mAV_calculation[2]
    
    def call_rsi(self, close_data, dates, start):
        sell_or_buy, c_data, rsi_balances, rsi_b_dates = [], [], [], []
        rsi_amount = 0
        rsi_balance = 100

        rsi_bORs = [[],[]]

        rsi_data = [[],[]]

        rsi_amount = (rsi_balance - 0.002*rsi_balance) / close_data[start]
        """ print("starting balance -> stochRSI: ", rsi_amount*close_data[start])
        print(dates[start], "  [rsi] buy at ", close_data[start]) """
        rsi_balances.append(rsi_amount*close_data[start])
        rsi_b_dates.append(dates[start])

        for day in range(start, len(close_data)+1):
            c_data = close_data[:day]
            d_data = dates[:day]
            
            sell_or_buy = self.simulate_rsi(c_data, d_data)  
            rsi_data[0].append(sell_or_buy[1]) 
            rsi_data[1].append(d_data[len(d_data)-1])

            rsi_bORs[0].append(sell_or_buy[0])
            rsi_bORs[1].append(d_data[len(d_data)-1])

            #rsi
            if sell_or_buy[0] == "buy" and rsi_amount == 0:
                rsi_amount = (rsi_balance - 0.002*rsi_balance) / c_data[len(c_data)-1]
                """ print(d_data[len(d_data)-1], "  [rsi] buy at ", c_data[len(c_data)-1]) """
            elif (sell_or_buy[0] == "sell" or day == len(close_data)) and rsi_amount != 0:
                rsi_balance = rsi_amount * c_data[len(c_data)-1]
                rsi_amount = 0
                rsi_balances.append(rsi_balance)
                rsi_b_dates.append(d_data[len(d_data)-1])
                """ print(d_data[len(d_data)-1], "  [rsi] sell at ", c_data[len(d_data)-1], "          ", rsi_balance) """

        return rsi_balances, rsi_b_dates, rsi_data, rsi_bORs

    def call_MFI(self, close_data, dates, high_data, low_data, volume_data, start):
        sell_or_buy, c_data, mfi_balances, mfi_b_dates = [], [], [], []
        mfi_amount = 0
        mfi_balance = 100

        mfi_bORs = [[],[]]

        cross = []

        mfi_amount = (mfi_balance - 0.002*mfi_balance) / close_data[start]
        print("starting balance -> mfi: ", mfi_amount*close_data[start])
        print(dates[start], "  [mfi] buy at ", close_data[start])
        mfi_balances.append(mfi_amount*close_data[start])
        mfi_b_dates.append(dates[start])

        for day in range(start, len(close_data)+1):
            c_data = close_data[:day]
            d_data = dates[:day]
            h_data = high_data[:day]
            l_data = low_data[:day]
            v_data = volume_data[:day]

            sell_or_buy = self.simulate_MFI(c_data, h_data, l_data, v_data, d_data) 

            mfi_bORs[0].append(sell_or_buy)
            mfi_bORs[1].append(d_data[len(d_data)-1])

            #mfi 
            if sell_or_buy[0] == "buy" and mfi_amount == 0:
                mfi_amount = (mfi_balance - 0.002*mfi_balance) / c_data[len(c_data)-1]
                print(d_data[len(d_data)-1], "  [mfi] buy at ", c_data[len(c_data)-1])
            if (sell_or_buy[0] == "sell" or day == len(close_data)) and mfi_amount != 0:
                mfi_balance = mfi_amount * c_data[len(c_data)-1]
                mfi_amount = 0
                mfi_balances.append(mfi_balance)
                mfi_b_dates.append(d_data[len(d_data)-1])
                print(d_data[len(d_data)-1], "  [mfi] sell at ", c_data[len(d_data)-1], "          ", mfi_balance)

        return mfi_balances, mfi_b_dates, mfi_bORs

    def call_mAV(self, close_data, dates, start):
        sell_or_buy, c_data, mAV_balances, mAV_b_dates = [], [], [], []
        mAV_amount = 0
        mAV_balance = 100

        mAV_bORs = [[],[]]

        cross = []

        mAV_amount = (mAV_balance - 0.002*mAV_balance) / close_data[start]
        """ print("starting balance -> mAV: ", mAV_amount*close_data[start])
        print(dates[start], "  [mAV] buy at ", close_data[start]) """
        mAV_balances.append(mAV_amount*close_data[start])
        mAV_b_dates.append(dates[start])

        for day in range(start, len(close_data)+1):
            c_data = close_data[:day]
            d_data = dates[:day]
            
            sell_or_buy = self.simulate_mAV(c_data, d_data)  

            mAV_bORs[0].append(sell_or_buy)
            mAV_bORs[1].append(d_data[len(d_data)-1])

            #mAV 
            if sell_or_buy == "buy" and mAV_amount == 0:
                mAV_amount = (mAV_balance - 0.002*mAV_balance) / c_data[len(c_data)-1]
                """ print(d_data[len(d_data)-1], "  [mAV] buy at ", c_data[len(c_data)-1]) """
            if (sell_or_buy == "sell" or day == len(close_data)) and mAV_amount != 0:
                mAV_balance = mAV_amount * c_data[len(c_data)-1]
                mAV_amount = 0
                mAV_balances.append(mAV_balance)
                mAV_b_dates.append(d_data[len(d_data)-1])
                """ print(d_data[len(d_data)-1], "  [mAV] sell at ", c_data[len(d_data)-1], "          ", mAV_balance) """

        return mAV_balances, mAV_b_dates, mAV_bORs

    def simulate_rsi(self, close_data, dates):
        rsi_buyOrSell = None

        #rsi
        rsi = self.rsi(8, close_data)
        stoch_rsi = self.stoch_rsi(8, rsi, 3)

        rsi_buyOrSell = self.buy_or_sell(stoch_rsi, 0.5, 0.1)

        return rsi_buyOrSell

    def simulate_MFI(self, close_data, high_data, low_data, volume_data, dates):
        mfi_buyOrSell = None
        period = 14

        #mfi
        mfi = self.calculate_MFI(high_data, low_data, close_data, volume_data, period, dates)

        mfi_buyOrSell =  self.buy_or_sell_MFI(mfi, 20,80, dates)

        return mfi_buyOrSell

    def simulate_mAV(self, close_data, dates):
        mAV_buyOrSell = None
        
        #average
        averageShort = self.moving_average(close_data, 2)
        averageLong = self.moving_average(close_data, 21)

        cross = self.ma_cross(averageShort, averageLong)
        mAV_buyOrSell = cross[len(cross)-1]

        return mAV_buyOrSell

    def combined(self, close_data, dates, rsi_calculation, mAV_calculation, start, end, low_data, high_data):
        """
        combines RSI and mAV
        """

        self.balance = 100
        self.amount = 0
        rsi_index, mAV_index = 0, 0
        balance_combined = [[],[]]
        date = 0

        lastBuyPrice = 0

        print("")
        print("")
        print("test with combined rsi and mAV:")

        self.amount = (self.balance - 0.002*self.balance) / close_data[start]
        balance_combined[0].append(self.balance)
        balance_combined[1].append(dates[start])
        self.balance = 0

        stopLoss = 0.045


        # print("starting balance: ", self.amount*close_data[start])
        # print(dates[start], "  buy at ", close_data[start])
        for idx in range(start, end):
            date = dates[idx]
            """ if high_data[idx] > lastBuyPrice:
                lastBuyPrice = high_data[idx] """

            try:
                rsi_index = rsi_calculation[1].index(date)
            except ValueError:
                rsi_index = 0

            try:
                mav_index = mAV_calculation[1].index(date)
            except ValueError:
                mav_index = 0

            # try:
            #     rsi_index = rsi_calculation[1].index(date)
            #     mAV_index = mAV_calculation[1].index(date)
            #     # av_i = av[1].index(date)
            # except ValueError:
            #     pass  # do nothing!

            #if rsi_calculation[0][rsi_index] == "buy" and self.amount == 0 and av[0][av_i] > av[0][av_i-1]:
            if rsi_calculation[0][rsi_index] == "buy" and self.amount == 0:
                self.amount = (self.balance - 0.002*self.balance) / close_data[idx]
                lastBuyPrice = close_data[idx]
                print(dates[idx], "  buy at ", close_data[idx], " high would be: ", high_data[idx], " low would be: ", low_data[idx])


            # elif (mAV_calculation[0][mAV_index] == "sell" or dates[idx] == dates[len(dates)-1]) and self.amount != 0:
            elif (mAV_calculation[0][mAV_index] == "sell" or dates[idx] == dates[len(dates)-1] or ((lastBuyPrice/close_data[idx])-1) > stopLoss ) and self.amount != 0:
                if ((lastBuyPrice/close_data[idx])-1) > stopLoss:
                    self.balance = self.amount * (lastBuyPrice-lastBuyPrice*stopLoss)
                else:
                    self.balance = self.amount * close_data[idx]
                self.amount = 0
                balance_combined[0].append(self.balance)
                balance_combined[1].append(dates[idx])
                print(dates[idx], "  sell at ", close_data[idx], " high would be: ", high_data[idx], " low would be: ", low_data[idx], "          ", self.balance)

        return balance_combined

    def combinedMFIold(self, close_data, dates, rsi_calculation, mav_calculation, start, end, low_data, high_data, mfi):
        """
        combines RSI and MFI
        """

        self.balance = 100
        self.amount = 0
        rsi_index, mfi_index = 0, 0
        balance_combined = [[],[]]
        date = 0
        lastBuyPrice = 0

        print("")
        print("")
        print("test with combined rsi and mfi:")
        print("")

        self.amount = (self.balance - 0.002*self.balance) / close_data[start]
        balance_combined[0].append(self.balance)
        balance_combined[1].append(dates[start])
        self.balance = 0
        
        stopLoss = 0.05

        print("starting balance: ", self.amount*close_data[start])
        print(dates[start], "  buy at ", close_data[start])
        
        for idx in range(start, end-1):
            date = dates[idx]

            try:
                rsi_index = rsi_calculation[1].index(date)
                mav_index = mav_calculation[1].index(date)
                mfi_i = mfi[1].index(date)
            except ValueError:
                pass  # do nothing!
            
            """ if high_data[idx] > lastBuyPrice:
                lastBuyPrice = high_data[idx] """

            #if (mfi[0][mfi_i] == "buy" or rsi_calculation[0][rsi_index] == "buy" ) and self.amount == 0 and av[0][av_i] > av[0][av_i-1]:
            if (rsi_calculation[0][rsi_index] == "buy" ) and self.amount == 0:
                self.amount = (self.balance - 0.002*self.balance) / close_data[idx]
                lastBuyPrice = close_data[idx]
                print(dates[idx], "  buy at ", close_data[idx], " high would be: ", high_data[idx], " low would be: ", low_data[idx])


            # elif (dates[idx] == dates[len(dates)-1] or mfi[0][mfi_i] == "sell") and self.amount != 0:
            elif (mav_calculation[0][mav_index] == "sell" or dates[idx] == dates[len(dates)-1] or mfi[0][mfi_i] == "sell") and self.amount != 0:
            # elif (mav_calculation[0][mav_index] == "sell" or mfi[0][mfi_i] == "sell" or dates[idx] == dates[len(dates)-1] or ((lastBuyPrice/low_data[idx])-1) > stopLoss ) and self.amount != 0:

                """ if ((lastBuyPrice/low_data[idx])-1) > stopLoss:
                    self.balance = self.amount * (lastBuyPrice-lastBuyPrice*stopLoss)
                    print(dates[idx], "  sell at ", (lastBuyPrice-lastBuyPrice*stopLoss), " high would be: ", high_data[idx], " low would be: ", low_data[idx], "          ", self.balance)
                else: """
                self.balance = self.amount * close_data[idx]
                print(dates[idx], "  sell at ", close_data[idx], " high would be: ", high_data[idx], " low would be: ", low_data[idx], "          ", self.balance)
                self.amount = 0
                balance_combined[0].append(self.balance)
                balance_combined[1].append(dates[idx])
                

        return balance_combined

    def combinedMFI(self, close_data, dates, rsi_calculation, mav_calculation, start, end, low_data, high_data, mfi, open_data, resistances, supports, rsi_mid, mfi_mid, mid_open, mid_dates, mid_low):
        """
        combines RSI and MFI
        """
        countBuy, counter = 0,0

        self.balance = 100
        self.amount = 0
        midAmount = 0
        balance_combined = [[],[]]
        date = 0
        rsi_index, mav_index, mfi_i = 0, 0, 0

        balance_combined[0].append(self.balance)
        balance_combined[1].append(dates[start])
        
        lastBuyPriceMid = 0
        stopLossMid = 0.042
        lastBuyPrice = 0
        stopLoss = 0.042

        buyEnabled = True
        enable30min = True
        bought = 2
        sold = 2
        for idx in range(0, len(rsi_calculation[0])-2):
            date = dates[start+idx]
            rsi_index, mav_index, mfi_i = idx, idx, idx

            priceBuy = open_data[start + idx + 0] #if open is rsi source then buy at open of same candle 
            price = open_data[start + idx + 1] #if close is source of indicator then buy at next open

            for num, dat in enumerate(resistances[0][1], start = 0):
                if dat==date:
                    resistanceIdx = num-1

            if mav_calculation[0][mav_index] == "buy":
                buyEnabled = True

################# short time trader ###########

            if enable30min and buyEnabled:
                # handle index of 30m rsi
                try:    
                    mid_idx = rsi_mid[1].index(date)
                except ValueError:
                    mid_idx = 0

                if not mid_idx == 0:
                    for i in range(0,6):

                        midPrice = mid_open[mid_idx] # <- for rsi
                        # midPrice = mid_open[mid_idx+1] # <- for mfi
                        # print("rsi ", rsi_mid[1][mid_idx], rsi_mid[2][mid_idx], "mfi: ", mfi_mid[1][mid_idx], mfi_mid[2][mid_idx] )

                        # print("rsi ", rsi_mid[1][mid_idx], rsi_mid[2][mid_idx], midPrice)

                        if rsi_mid[0][mid_idx] == "buy" and mfi_mid[0][mid_idx] == "buy" and self.getAvailBalance() >= 20.0: 
                        # if mfi_mid[0][mid_idx] == "buy" and self.getAvailBalance() >= 20.0:
                        # if mfi_mid[2][mid_idx] > mfi_mid[2][mid_idx-1] and self.getAvailBalance() >= 20.0: #and rsi_mid[2][mid_idx+1] > rsi_mid[2][mid_idx] and self.getAvailBalance() >= 20.0:
                            midAmount += self.calcMaxAmount(midPrice)/2
                            self.simulate_buy(self.calcMaxAmount(midPrice)/2, midPrice)
                            lastBuyPrice = midPrice
                            lastBuyPriceMid = mid_low[mid_idx]
                            balance_combined[0].append(self.getWalletBalance(midPrice))
                            balance_combined[1].append(mid_dates[mid_idx])
                            print(mid_dates[mid_idx], " RSI - MID buy at open ", midPrice)

                        elif (rsi_mid[0][mid_idx] == "sell"
                            or (i == 0 and rsi_calculation[0][rsi_index] == "sell")
                            or ((lastBuyPriceMid/midPrice)-1) > stopLoss
                            ) and self.getAvailAmount() > 0.0:
                        # elif mfi_mid[0][mid_idx] == "sell" and self.getAvailAmount() > 0.0:
                        # elif mfi_mid[2][mid_idx] < mfi_mid[2][mid_idx-1] and self.getAvailAmount() > 0.0: #and rsi_mid[2][mid_idx+1] < rsi_mid[2][mid_idx] and self.getAvailAmount() > 0.0:

                            if (i == 0 and rsi_calculation[0][rsi_index] == "sell"):
                                print(mid_dates[mid_idx], " RSI - 3H sell at open ", priceBuy, "              ", self.getWalletBalance(midPrice))
                                self.simulate_sell(midAmount, priceBuy)
                                midAmount = 0
                            elif ((lastBuyPriceMid/midPrice)-1) > stopLoss:
                                print(mid_dates[mid_idx], " STOPLLOSS - MID sell at open ", midPrice, "              ", self.getWalletBalance(midPrice))
                                self.simulate_sell(self.getAvailAmount()/2, midPrice)
                                midAmount -= self.getAvailAmount()/2
                            else:
                                print(mid_dates[mid_idx], " RSI - MID sell at open ", midPrice, "              ", self.getWalletBalance(midPrice))
                                self.simulate_sell(self.getAvailAmount()/2, midPrice)
                                midAmount -= self.getAvailAmount()/2
                                # self.simulate_sell(self.getAvailAmount()*(3/4), midPrice)
                                # midAmount -= self.getAvailAmount()*(3/4)

                            balance_combined[0].append(self.getWalletBalance(midPrice))
                            balance_combined[1].append(mid_dates[mid_idx])

                            print(mid_dates[mid_idx], " RSI - MID sell at open ", midPrice, "              ", self.getWalletBalance(midPrice))

                        mid_idx += 1
                else:
                    print("an error occured finding the current 30m index!")

################ long time trader ########

            if buyEnabled and (rsi_calculation[0][rsi_index] == "buy" 
                or priceBuy > resistances[0][0][resistanceIdx] 
                or priceBuy > resistances[1][0][resistanceIdx] 
                or priceBuy > resistances[2][0][resistanceIdx]
                ) and self.getAvailBalance() >= 20.0 and sold > 1:

                enable30min = False
                bought = 0

                if (rsi_calculation[0][rsi_index] == "buy"
                    or priceBuy > resistances[0][0][resistanceIdx] 
                    or priceBuy > resistances[1][0][resistanceIdx] 
                    or priceBuy > resistances[2][0][resistanceIdx]
                    ):
                    
                    # self.simulate_buy(self.calcMaxAmount(priceBuy)*(2/3), priceBuy)
                    # lastBuyPrice = priceBuy
                    lastBuyPrice = low_data[start + idx + 0]
                    counter = 0
                    self.simulate_buy(self.calcMaxAmount(priceBuy), priceBuy)

                else:
                    counter = counter + 1
                    self.simulate_buy(self.calcMaxAmount(priceBuy)/(5 + counter * -1), priceBuy)
                    print(dates[start+idx], "Buy at ", priceBuy ," 1/", 5 + counter * -1, " curr rsi: ",  rsi_calculation[2][rsi_index], " prev rsi: ", rsi_calculation[2][rsi_index-1])
                    if counter >= 4:
                        counter = 0

                balance_combined[0].append(self.getWalletBalance(priceBuy))
                balance_combined[1].append(dates[start+idx])

                if rsi_calculation[0][rsi_index] == "buy":
                    print(dates[start+idx], " RSI ", rsi_calculation[2][rsi_index]," buy at open ", priceBuy, " high would be: ", high_data[start+idx-1], " low would be: ", low_data[start+idx-1])
                if priceBuy > resistances[0][0][resistanceIdx] or priceBuy > resistances[1][0][resistanceIdx] or priceBuy > resistances[2][0][resistanceIdx]:
                    print(dates[start+idx], " RESISTANCE buy at open ", priceBuy, " high would be: ", high_data[start+idx-1], " low would be: ", low_data[start+idx-1])

            elif (mav_calculation[0][mav_index] == "sell" 
                or mfi[0][mfi_i] == "sell" 
                or dates[start+idx+1] == dates[len(dates)-2] 
                or ((lastBuyPrice/price)-1) > stopLoss # <-- stoploss
                or price < supports[0][0][resistanceIdx] 
                or price < supports[1][0][resistanceIdx] 
                or price < supports[2][0][resistanceIdx]
                ) and self.getAvailAmount() > 0.0 and bought > 1: 
                sold = 0
                enable30min = True

                if (mav_calculation[0][mav_index] == "sell" 
                    or mfi[0][mfi_i] == "sell" 
                    or dates[start+idx+1] == dates[len(dates)-2] 
                    or open_data[start + idx+1] < supports[0][0][resistanceIdx] 
                    or open_data[start + idx+1] < supports[1][0][resistanceIdx] 
                    or open_data[start + idx+1] < supports[2][0][resistanceIdx]
                    or ((lastBuyPrice/price)-1) > stopLoss
                    ):
                    
                    # self.simulate_sell(self.getAvailAmount()*(2/3), price)
                # elif ((lastBuyPrice/price)-1) > stopLoss:
                    self.simulate_sell(self.getAvailAmount(), price)
                else:
                    counter = counter + 1
                    self.simulate_sell(self.getAvailAmount()/(5 + counter * -1), price)
                    print("Sell at ", price ," 1/", 5 + counter * -1)
                    if counter >= 4:
                        counter = 0

                balance = self.getWalletBalance(price)

                if ((lastBuyPrice/price)-1) > stopLoss:
                    buyEnabled = False
                    print(dates[start+idx+1], " STOPLOSS sell at open", price, " high would be: ", high_data[start+idx], " low would be: ", low_data[start + idx], "          ", balance)
                else:
                    if(price < supports[0][0][resistanceIdx] or price < supports[1][0][resistanceIdx] or price < supports[2][0][resistanceIdx]):
                        print(dates[start+idx+1], " Resistance sell at open", open_data[start + idx+1], " high would be: ", high_data[start+idx], " low would be: ", low_data[idx], "          ", balance)
                    if(mav_calculation[0][mav_index] == "sell"):
                        print(dates[start+idx+1], " MAV sell at open", price, " high would be: ", high_data[start+idx], " low would be: ", low_data[idx], "          ", balance)
                    if(mfi[0][mfi_i] == "sell"):
                        print(dates[start+idx+1], " MFI sell at open", price, " high would be: ", high_data[start+idx], " low would be: ", low_data[start+idx], "          ", balance)
                    if(dates[start+idx+1] == dates[len(dates)-2]):
                        print(dates[start+idx+1], " FIN sell at open", price, " high would be: ", high_data[start+idx], " low would be: ", low_data[start+idx], "          ", balance)
                print("")
                
                balance_combined[0].append(self.getWalletBalance(price))
                balance_combined[1].append(dates[start+idx+1])

            bought += 1
            sold += 1

        profitable = 0
        for i in range(2,len(balance_combined[0])):
            if balance_combined[0][i] > balance_combined[0][i-1]:
                profitable +=1
        
        profTrades = 0
        if len(balance_combined[0])-2 > 0:
            profTrades = profitable/(len(balance_combined[0])-2)
            print("Profitable trades: ", round(100*profTrades,2), "%")

        print("buy and hold balance: ",((100 - 0.002*100) / open_data[start]) * open_data[len(open_data)-1])
        print("\n\n\n")
        return balance_combined, profTrades

    def simulate_buy(self, amount, price):
        if(self.getAvailBalance() > 0 and self.getAvailBalance() >= (amount * price / 0.998)):
            self.setAvailBalance(round(self.getAvailBalance() - (amount * price / 0.998),5))
            self.setAvailAmount(self.getAvailAmount() + amount)
        else:
            self.setAvailAmount(self.getAvailAmount() + self.calcMaxAmount(price))
            self.setAvailBalance(0.0)

    def simulate_sell(self, amount, price):
        if(self.getAvailAmount() > 0 and amount <= self.getAvailAmount()):
            self.setAvailBalance(self.getAvailBalance() + amount * price)
            self.setAvailAmount(self.getAvailAmount() - amount)
        else:
            self.setAvailBalance(self.getAvailBalance() + self.getAvailAmount() * price)
            self.setAvailAmount(0.0)

    def calcMaxAmount(self, price):
        return round(((0.998 * self.getAvailBalance()) / price), 5)
    
    def getWalletBalance(self, price):
        return round(self.getAvailBalance() + self.getAvailAmount() * price,5)

    def getAvailAmount(self):
        return self.amount

    def setAvailAmount(self, amount):
        self.amount = amount

    def getAvailBalance(self):
        return self.balance

    def setAvailBalance(self, balance):
        self.balance = balance

class Helper:
    def __init__(self, key, secret):
        self.BASE_URL = "https://api.bitfinex.com/"
        #self.URL = "{0:s}://{1:s}/{2:s}".format(PROTOCOL, HOST, VERSION2)
        self.KEY = key
        self.SECRET = secret
        pass
    
    def convert_scrapedCandles(self):
        dates, close_data, low_data, high_data, open_data, volume_data = [], [], [], [], [], []

        # open the file in read binary mode
        file = open("candles30m_2017:09:01", "rb")
        #read the file to numpy array
        candles = np.load(file)

        for candle in candles:
            ts = int(candle[0]) / 1000 
            dates.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
            open_data.append(candle[1])
            high_data.append(candle[3])
            low_data.append(candle[4])
            close_data.append(candle[2])
            volume_data.append(abs(candle[5]))

        dates = list(reversed(dates))
        open_data = list(reversed(open_data))
        close_data = list(reversed(close_data))
        high_data = list(reversed(high_data))
        low_data = list(reversed(low_data))
        volume_data = list(reversed(volume_data))

        return dates, open_data, close_data, high_data, low_data, volume_data

    def getCandles(self, timeFrame, symbol, parameters=None):
        dates, close_data, low_data, high_data, open_data, volume_data = [], [], [], [], [], []

        # set the parameters to limit the number of bids or asks
        if parameters == None:
            parameters = {'limit': 10000, 'sort': -1}

        timeFrame_symbol = ":" + timeFrame + ":t" + symbol
        if timeFrame == "3h":
            candles = self._get_3h(self.url_for(PATH_CANDLES+'/hist', (timeFrame_symbol), parameters=parameters, version=VERSION2))
        else:
            candles = self._get(self.url_for(PATH_CANDLES+'/hist', (timeFrame_symbol), parameters=parameters, version=VERSION2))

        for candle in candles:
            ts = int(candle[0]) / 1000 
            dates.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
            open_data.append(candle[1])
            close_data.append(candle[2])
            high_data.append(candle[3])
            low_data.append(candle[4])
            volume_data.append(candle[5])

        dates = list(reversed(dates))
        open_data = list(reversed(open_data))
        close_data = list(reversed(close_data))
        high_data = list(reversed(high_data))
        low_data = list(reversed(low_data))
        volume_data = list(reversed(volume_data))

        return dates, open_data, close_data, high_data, low_data, volume_data

    @retry(wait=wait_fixed(15))
    def _get_3h(self, url):
        try:
            data = requests.get(url, timeout=TIMEOUT)
            with open('btcusd-3h.json', 'w+') as outf:
             outf.write(data.text)
             outf.close()
            return data.json()
            # return requests.get(url, timeout=TIMEOUT).json()
        except requests.exceptions.Timeout:
            print("timeout error occurred")
            raise TimeoutError
        except requests.exceptions.RequestException as e:
            with open('btcusd-3h.json', 'r') as file:
             return json.load(file)

    @retry(wait=wait_fixed(15))
    def _get(self, url):
        try:
            data = requests.get(url, timeout=TIMEOUT)
            with open('btcusd-30m.json', 'w+') as outf:
             outf.write(data.text)
             outf.close()
            return data.json()
            # return requests.get(url, timeout=TIMEOUT).json()
        except requests.exceptions.Timeout:
            print("timeout error occurred")
            raise TimeoutError
        except requests.exceptions.RequestException as e:#
            with open('btcusd-30m.json', 'r') as file:
             return json.load(file)
            print("catastrophic error. bail.")
            # raise SystemExit(e)
        
    def server(self, version=VERSION):
        return u"{0:s}://{1:s}/{2:s}".format(PROTOCOL, HOST, version)

    def _build_parameters(self, parameters):
        # sort the keys so we can test easily in Python 3.3 (dicts are not
        # ordered)
        keys = list(parameters.keys())
        keys.sort()

        return '&'.join(["%s=%s" % (k, parameters[k]) for k in keys])

    def url_for(self, path, path_arg=None, parameters=None, version=VERSION):

        # build the basic url
        url = "%s/%s" % (self.server(version), path)

        # If there is a path_arg, interpolate it into the URL.
        # In this case the path that was provided will need to have string
        # interpolation characters in it, such as PATH_TICKER
        if path_arg:
            url = url % (path_arg)

        # Append any parameters to the URL.
        if parameters:
            url = "%s?%s" % (url, self._build_parameters(parameters))

        return url

    def _convert_to_floats(self, data):
        """
        Convert all values in a dict to floats
        """
        for key, value in enumerate(data):
            data[key] = float(value)

        return data

    #####################################
    # helper for authenticated requests #
    #####################################

    def _headers(self, path, nonce, body):
        secbytes = self.SECRET.encode(encoding='UTF-8')
        signature = "/api/" + path + nonce + body
        sigbytes = signature.encode(encoding='UTF-8')
        h = hmac.new(secbytes, sigbytes, hashlib.sha384)
        hexstring = h.hexdigest()
        return {
            "bfx-nonce": nonce,
            "bfx-apikey": self.KEY,
            "bfx-signature": hexstring,
            "content-type": "application/json"
        }

    @retry(wait=wait_fixed(15))
    def req(self, path, params = {}):
        try:
            nonce = self._nonce()
            body = params
            rawBody = json.dumps(body)
            headers = self._headers(path, nonce, rawBody)
            url = self.BASE_URL + path
            resp = requests.post(url, headers=headers, data=rawBody, verify=True)
            return resp
        except requests.exceptions.Timeout:
            print("timeout error occurred")
            raise TimeoutError
        except Exception as e:
            print("catastrophic error. bail.")
            raise SystemExit(e)

    def _nonce(self):
        return str(int(round(time.time() * 1000000)))      
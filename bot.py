import ccxt
import ta
import pandas as pd
import math
#Bollinger Bands help identify sharp, short-term price movements and potential entry and exit points.
#ATR measures volatility, taking into account any gaps in the price movement. Typically, ATR calculation is based on 14 periods - can be intraday, daily, weekly, or monthly.
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator

#Sets up data & variables
exchange = ccxt.kraken()
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=270)
df = pd.DataFrame(bitcoin_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
bb_indicator = BollingerBands(df['close'],window=10)

#Just testing it works
#for candle in bitcoin_data:
    #print(candle)


#Adds upper band to dataframe
df['upper_band'] = bb_indicator.bollinger_hband()

#Adds lower band to dataframe 
df['lower_band'] = bb_indicator.bollinger_lband()

#Adds smooth moving average to dataframe
df['smooth moving_average']=bb_indicator.bollinger_mavg()

#Adds Exponential Moving Avergae to dataframe
df['EMA'] = EMAIndicator(df['close'], window = 5).ema_indicator()

#Adds average true range indicator to dataframe
atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'])
df['atr'] = atr_indicator.average_true_range()

#Adds Simple Moving Average to the dataframe
df['SMA'] = SMAIndicator(df['close'],window=5).sma_indicator()

#Simplifies the timestamp to be easier to use with triggers
df['simplified_timestamp'] = pd.to_numeric((df['timestamp'] - df['timestamp'].loc[0])  / (df['timestamp'].loc[2] - df['timestamp'].loc[1]), downcast='signed')

#Just testing it works
#print(df.rows)


#Buy Trigger to return true or false if we should buy on a particular timestamp according to our parameters
def buyTrigger(timestamp, parameters=0):
    return buy(timestamp, parameters) and (not buy(timestamp-1, parameters)) and (not (sell(timestamp, parameters) and not sell(timestamp-1, parameters)))

#Sell Trigger to return true or false if we should sell on a particular timestamp according to our parameters
def sellTrigger(timestamp, parameters=0):
    return sell(timestamp, parameters) and (not sell(timestamp-1, parameters)) and (not (buy(timestamp, parameters) and not buy(timestamp-1, parameters)))

#BUY AND SELL FUNCTIONS JUST FOR TESTING NOW TO SEE IF TRIGGERS WORK, NEED TO FILL IN WITH ACTUAL ALGORITHM LATER
def buy(timestamp, parameters):
    #PLAYING AROUND WITH EMA AND SMA
    '''EMA = df[df['simplified_timestamp'] == timestamp]['EMA'][timestamp]
    SMA = df[df['simplified_timestamp'] == timestamp]['SMA'][timestamp]
    if math.isnan(EMA) or math.isnan(SMA):
        return False
    if EMA > SMA:
        return True
    return False'''
    lower = df[df['simplified_timestamp'] == timestamp]['lower_band'][timestamp]
    close = df[df['simplified_timestamp'] == timestamp]['close'][timestamp]
    if close < lower:
        return True
    return False

def sell(timestamp, parameters):
    #Testing with EMA and SMA
    '''EMA = df[df['simplified_timestamp'] == timestamp]['EMA'][timestamp]
    SMA = df[df['simplified_timestamp'] == timestamp]['SMA'][timestamp]
    if math.isnan(EMA) or math.isnan(SMA):
        return False
    if EMA < SMA:
        return True
    return False'''

    upper = df[df['simplified_timestamp'] == timestamp]['upper_band'][timestamp]
    close = df[df['simplified_timestamp'] == timestamp]['close'][timestamp]
    if upper < close:
        return True
    return False

def evaluate():
    money = 100
    bitcoin = 0
    for row in range(len(df)):
        if buyTrigger(row):
            bitcoin += money / df['close'].loc[row]
            money = 0
        elif row == len(df)-1:
            money += bitcoin * df['close'].loc[row]
        elif sellTrigger(row):
            money += bitcoin * df['close'].loc[row]
            bitcoin = 0
        print(row, "Money:", money, "Bitcoin:", bitcoin)
    return money


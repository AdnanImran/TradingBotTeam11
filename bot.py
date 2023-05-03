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
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=720)
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
def buyTrigger(timestamp, parameters):
    return buy(timestamp, parameters) and (not buy(timestamp-1, parameters)) and (not (sell(timestamp, parameters) and not sell(timestamp-1, parameters)))

#Sell Trigger to return true or false if we should sell on a particular timestamp according to our parameters
def sellTrigger(timestamp, parameters):
    return sell(timestamp, parameters) and (not sell(timestamp-1, parameters)) and (not (buy(timestamp, parameters) and not buy(timestamp-1, parameters)))

#BUY AND SELL FUNCTIONS JUST FOR TESTING NOW TO SEE IF TRIGGERS WORK, NEED TO FILL IN WITH ACTUAL ALGORITHM LATER
def buy(timestamp, df):
    #PLAYING AROUND WITH EMA AND SMA
    '''EMA = df[df['simplified_timestamp'] == timestamp]['EMA'][timestamp]
    SMA = df[df['simplified_timestamp'] == timestamp]['SMA'][timestamp]
    if math.isnan(EMA) or math.isnan(SMA):
        return False
    if EMA > SMA:
        return True
    return False'''
    trigger = df[df['simplified_timestamp'] == timestamp]["Indicator"][timestamp]
    lower = df[df['simplified_timestamp'] == timestamp]['lower'][timestamp]
    if trigger < lower:
        return True
    return False

def sell(timestamp, df):
    #Testing with EMA and SMA
    '''EMA = df[df['simplified_timestamp'] == timestamp]['EMA'][timestamp]
    SMA = df[df['simplified_timestamp'] == timestamp]['SMA'][timestamp]
    if math.isnan(EMA) or math.isnan(SMA):
        return False
    if EMA < SMA:
        return True
    return False'''

    upper = df[df['simplified_timestamp'] == timestamp]['upper'][timestamp]
    trigger = df[df['simplified_timestamp'] == timestamp]["Indicator"][timestamp]
    if upper < trigger:
        return True
    return False

# PARAMETERS = [ TRIGGER_INDICATOR (SMA=0, CLOSE=1, EMA=2), BOLLINGER_BANDS_WINDOW, TRIGGER_WINDOW_SIZE ] 
def evaluate(parameters=[0,14,3]):
    #initialise temp dataframe
    eval_df = pd.DataFrame()
    eval_df["simplified_timestamp"] = df["simplified_timestamp"]
    BOLLINGER_BANDS_WINDOW = parameters[1]
    TRIGGER_WINDOW_SIZE = parameters[2]
    #FIND TRIGGER INDICATOR
    if parameters[0] == 0: #SMA
        eval_df["Indicator"] = SMAIndicator(df['close'],window=TRIGGER_WINDOW_SIZE).sma_indicator()
    elif parameters[0] == 1: #CLOSE
        eval_df["Indicator"] = df['close']
    elif parameters[0] == 2: #EMA
        eval_df["Indicator"] = EMAIndicator(df['close'], window = TRIGGER_WINDOW_SIZE).ema_indicator()
    
    #BOLLINGER BANDS
    bands = BollingerBands(df['close'],window=BOLLINGER_BANDS_WINDOW)
    eval_df["upper"] = bands.bollinger_hband()
    eval_df["lower"] = bands.bollinger_lband()
    
    money = 100
    bitcoin = 0
    for row in range(len(df)):
        if buyTrigger(row, eval_df):
            bitcoin += money / df['close'].loc[row]
            money = 0
        elif row == len(df)-1:
            money += bitcoin * df['close'].loc[row]
        elif sellTrigger(row, eval_df):
            money += bitcoin * df['close'].loc[row]
            bitcoin = 0
        print(row, "Money:", money, "Bitcoin:", bitcoin)
    return money


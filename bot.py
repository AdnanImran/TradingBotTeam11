import ccxt
import ta
import pandas as pd
#Bollinger Bands help identify sharp, short-term price movements and potential entry and exit points.
#ATR measures volatility, taking into account any gaps in the price movement. Typically, ATR calculation is based on 14 periods - can be intraday, daily, weekly, or monthly.
from ta.volatility import BollingerBands, AverageTrueRange 

#Sets up data & variables
exchange = ccxt.kraken()
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=270)
df = pd.DataFrame(bitcoin_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
bb_indicator = BollingerBands(df['close'])

#Just testing it works
#for candle in bitcoin_data:
    #print(candle)


#Adds upper band to dataframe
df['upper_band'] = bb_indicator.bollinger_hband()

#Adds lower band to dataframe 
df['lower_band'] = bb_indicator.bollinger_lband()

#Adds moving average to dataframe
df['moving_average']=bb_indicator.bollinger_mavg()

#Adds average true range indicator to dataframe
atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'])
df['atr'] = atr_indicator.average_true_range()

#Just testing it works
#print(df)
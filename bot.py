import ccxt
import ta

exchange = ccxt.kraken()
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=270)

#Just testing it works
#for candle in bitcoin_data:
    #print(candle)
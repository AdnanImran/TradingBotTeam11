import ccxt
import ta
import pandas as pd
#Bollinger Bands help identify sharp, short-term price movements and potential entry and exit points.
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator

#Sets up data & variables
exchange = ccxt.kraken()
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=720)
df = pd.DataFrame(bitcoin_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
bb_indicator = BollingerBands(df['close'],window=5)

#Adds upper band to dataframe
df['upper_band'] = bb_indicator.bollinger_hband()

#Adds lower band to dataframe 
df['lower_band'] = bb_indicator.bollinger_lband()

#Adds moving average to dataframe
df['moving_average']=bb_indicator.bollinger_mavg()

#Adds average true range indicator to dataframe
atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'])
df['atr'] = atr_indicator.average_true_range()

#Adds Simple Moving Average to the dataframe
df['SMA'] = SMAIndicator(df['close'],window=5).sma_indicator()

#Optimization Algorithm
def optimize():
    """Typical parameters used in DE are listed as following:
    D – Problem dimension -> Defined by the number of decision variables or parameters that are being optimized in the fitness function. 2 (or 3 - will we include fillna?)
    N – Population size (pop_var) -> How many input variables will be consider each time?
    Cr – Crossover probability (cross_prob) -> value between 0 and 1. 0 means there will be no crossover. 
    #The higher the value the more the algorithm will explore. The lower the value the more the algorithm can refine promising solution spaces. 
    #Can we do a high value first and then create a list of optimal solutions and run it once more and lower the value?
    F – Scaling factor ->  Controls the amplification of the difference vector (difference between two randomly selected individuals from the population) used in the mutation step. 
    G – Number of generation/stopping condition -> Decide how many iterations should be considered.
    Li,Hi – boundary for dimension i -> the range of values that 'dimension i' of each candidate solution is allowed to take during the optimization process. These boundaries help to constrain the search space of the algorithm.
    #Basically it helps ensure that candidate solutions remain within the feasible region of the search space.
    """
    return None

#Tells the DE algorithm if a solution is viable or not.
def fitness(candidate_solution, df):
    return None


#Buy Trigger to return true or false if we should buy on a particular timestamp according to our parameters
def buyTrigger(timestamp, parameters):
    return buy(timestamp, parameters) and (not buy(timestamp-1, parameters)) and (not (sell(timestamp, parameters) and not sell(timestamp-1, parameters)))

#Sell Trigger to return true or false if we should sell on a particular timestamp according to our parameters
def sellTrigger(timestamp, parameters):
    return sell(timestamp, parameters) and (not sell(timestamp-1, parameters)) and (not (buy(timestamp, parameters) and not buy(timestamp-1, parameters)))

#BUY AND SELL FUNCTIONS JUST FOR TESTING NOW TO SEE IF TRIGGERS WORK, NEED TO FILL IN WITH ACTUAL ALGORITHM LATER
def buy(timestamp, parameters):
    if timestamp == 0:
        return False
    return True

def sell(timestamp, parameters):
    if timestamp == 1:
        return False
    return True



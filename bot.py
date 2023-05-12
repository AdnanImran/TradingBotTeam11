import ccxt
import ta
import pandas as pd
import random
import math
import numpy
#Bollinger Bands help identify sharp, short-term price movements and potential entry and exit points.
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator

#Sets up data & variables
exchange = ccxt.kraken()
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=720)

# split ratio in range (0,1]. default = 0.8 (80% training, 20% testing). Set to 1 to test the whole 2 year period against etherium
split_ratio = 0.8
test_at_start = False  # Should test data be start or end of 2-year period?

split = round(split_ratio*720)

if split_ratio == 1:
    print("Forward testing against etherium")
else:
    print(f"Split: {round(split_ratio*100)}% training, {round((1-split_ratio)*100)}% testing")

# Testing data at start of period
if test_at_start:
    df_train = pd.DataFrame(bitcoin_data[720-split:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_test = pd.DataFrame(bitcoin_data[:720-split], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Testing data at end of period
else:
    df_train = pd.DataFrame(bitcoin_data[:split], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_test = pd.DataFrame(bitcoin_data[split:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


# Assign df to training dataframe initially
df = df_train 

bb_indicator = BollingerBands(df['close'],window=5)

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

#Simplifies the timestamp to be easier to use with triggers
df['simplified_timestamp'] = pd.to_numeric((df['timestamp'] - df['timestamp'].loc[0])  / (df['timestamp'].loc[2] - df['timestamp'].loc[1]), downcast='signed')

#Optimization Algorithm
def optimize(buyLimit, hyperparameters = [10,0.7,0.5,20]):
    popSize = hyperparameters[0]  # 10 Population size -> How many versions of parameters will be created in each generation
    recombinationValue = hyperparameters[1] # 0.7
    mutationValue = hyperparameters[2] # 0.5 Scaling factor ->  Controls the amplification of the difference vector (difference between two randomly selected individuals from the population) used in the mutation step.
    gen = hyperparameters[3]  # 20 Number of generation/stopping condition -> Decide how many iterations should be considered.     
    bounds = [(0,2),(1,buyLimit),(1,100),(0,3)] #Li,Hi – boundary for dimension i -> These boundaries help to constrain the search space of the algorithm.

    # Initialise Population and randomly pick values for the parameters
    population = initPopulation(popSize, bounds)
    
    # Initialise Lists for keeping track of results
    best_solution = []
    average_solution = []
     # Go through each generation
    for i in range(1,gen+1):
        print('GENERATION:',i)

        # Keep track of each generations score
        gen_scores = [] 
        # Go through each individual in the population
        for j in range(0, popSize):
            #--- MUTATION -----
            # Select three random vector index positions [0, popSize), not including current vector (j)
            # from the current generation that are both unique to themselves, but also unique to the 
            # currently selected individual that we're mutating (j)
            candidates = list(range(popSize))
            candidates.remove(j)
            # Random.sample method handles the unique selection
            # Parameters are the population and number of index positions to select
            random_index = random.sample(candidates, 3)

            # Get the values from the actual population
            x1 = population[random_index[0]]
            x2 = population[random_index[1]]
            x3 = population[random_index[2]]
            xT = population[j]

            # NOTE: Mutation strategy taken from this article: 
            # "Quantitative Trading Machine Learning Using Differential Evolution Algorithm" by Napas Vinitnantharat, Narit Inchan, 
            # Thatthai Sakkumjorn, Kitsada Doungjitjaroen & Chukiat Worasucheep
            # The mutation formula is as follows v = x1 + F(x2-x3) where F is the mutation value
            
            # Subtract x3 from x2, and create a new vector (x_diff)
            # zip method will iterate through tuples. 
            # E.g. if a = (1,2) and b = (3,4) --> looping through zip(a,b) will yield (1,3) and (2,4).
            # Thus making it easier to get difference of a certain parameter
            xDiff = []
            for x2_i, x3_i in zip(x2, x3):
                diff = x2_i - x3_i
                xDiff.append(diff)

            # multiply xDiff by the mutation factor (F) and add to x1
            vectorDonor = []
            for x1_i, xDiff_i in zip(x1, xDiff):
                v = x1_i + mutationValue * xDiff_i
                vectorDonor.append(v)
            
            # Checks that new values are within the bounds of our parameters
            vectorDonor = checkBounds(vectorDonor, bounds)

            #--- RECOMBINATION/CROSSOVER ----------
            # Recombination incorporates successful solutions from the previous generation
            # Here we randomly select which parameters should continue on, since this example only
            # has one parameter, it will randomly choose whether to take the parent or mutated child

            selected = []
            # cycle through each variable in our target vector
            for k in range(len(xT)):
                crossover = random.random()
                
                # recombination occurs when crossover <= recombination rate
                if crossover <= recombinationValue:
                    selected.append(vectorDonor[k])

                # recombination did not occur
                else:
                    selected.append(xT[k])
            #--- SELECTION ----------
            # This is a greedy method where if the new score is greater than previous score it will take new one
            # In this example scenario this is to max the area of a circle
            newScore  = trade(selected, buyLimit)
            oldScore = trade(xT, buyLimit)

            if newScore > oldScore:
                population[j] = selected
                gen_scores.append(newScore)

            else:
                gen_scores.append(oldScore)

        # Print each generations stats

        # current generation avg. fitness
        gen_avg = sum(gen_scores) / popSize
        # fitness of best individual
        gen_best = max(gen_scores)
        # solution of best individual
        gen_sol = population[gen_scores.index(max(gen_scores))]

        print('> GENERATION AVERAGE:',gen_avg)
        print('> GENERATION BEST:',gen_best)
        print('> BEST SOLUTION:',gen_sol,'\n')
        best_solution.append(gen_best)
        average_solution.append(gen_avg)
    print(gen_sol)
    return (gen_sol, best_solution, average_solution)

# Ensure that new values obtained through mutation are within the bounds of the parameters
def checkBounds(solutions, bounds):
    updatedSolutions = []
    # cycle through each variable in vector 
    for i in range(len(solutions)):

        # variable exceedes the minimum boundary
        if solutions[i] < bounds[i][0]:
            updatedSolutions.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if solutions[i] > bounds[i][1]:
            updatedSolutions.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= solutions[i] <= bounds[i][1]:
            updatedSolutions.append(solutions[i])
    return updatedSolutions

# Initialise the populationoptimize
# returns an array of parameter values for a solution.
# e.g. population = [ [0,1], [2,5] ]
def initPopulation(popSize, bounds):
    population = []
    for i in range(0,popSize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0],bounds[j][1]))
        population.append(indv)
    return population

# Tests the performance using backtesting the optimization algorithm. 
#NOTE:We could also split the data and do futuretesting. > Maybe it would be good to do this in a separate method.

def evaluate(results, buyLimit):
    #Compares optimized paramters against default parameters for bollinger bands
    #Default values 20 periods, 2 S.D.
    baseline=trade([1,20,2,2], buyLimit)
    print(f"Baseline: {baseline}")
    print(f"Results: {results}")
    successRate = ((results/baseline)-1)*100
    return successRate

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
def trade(parameters, buyLimit = 720,verbose=False):
    #initialise temp dataframe
    #print("Start Trade")
    eval_df = pd.DataFrame()
    eval_df["simplified_timestamp"] = df["simplified_timestamp"]
    BOLLINGER_BANDS_WINDOW = round(parameters[1])
    TRIGGER_WINDOW_SIZE = round(parameters[2])
    BOLLINGER_BANDS_WINDOW_DEV = round(parameters[3])
    #FIND TRIGGER INDICATOR
    if round(parameters[0]) == 0: #SMA
        eval_df["Indicator"] = SMAIndicator(df['close'],window=TRIGGER_WINDOW_SIZE).sma_indicator()
    elif round(parameters[0]) == 1: #CLOSE
        eval_df["Indicator"] = df['close']
    elif round(parameters[0]) == 2: #EMA
        eval_df["Indicator"] = EMAIndicator(df['close'], window = TRIGGER_WINDOW_SIZE).ema_indicator()
    
    #BOLLINGER BANDS
    bands = BollingerBands(df['close'],window=BOLLINGER_BANDS_WINDOW, window_dev=BOLLINGER_BANDS_WINDOW_DEV)
    eval_df["upper"] = bands.bollinger_hband()
    eval_df["lower"] = bands.bollinger_lband()
    #print( "Dataframe created")
    counter = 0
    money = 100
    bitcoin = 0
    for row in range(len(df)):
        if counter == buyLimit:
            if money == 0:
                money += bitcoin * df['close'].loc[row]
                bitcoin = 0
                counter = 0
            else:
                bitcoin += money / df['close'].loc[row]
                money = 0
                counter = 0
        elif bitcoin == 0:
            if buyTrigger(row, eval_df):
                bitcoin += money / df['close'].loc[row]
                money = 0 
        elif money == 0:
            if sellTrigger(row, eval_df):
                money += bitcoin * df['close'].loc[row]
                bitcoin = 0
        if row == len(df)-1:
            money += bitcoin * df['close'].loc[row]
        counter += 1
        if verbose:
            print(row, "Money:", money, "Bitcoin:", bitcoin)
    #print("End Train")
    return money



'''Three Key Functions'''
# Generate optimal parameters.
#buyLimit = 350
#tradeParameters = optimize(buyLimit)
#print(numpy.round(tradeParameters))
# Run the trade.
#results = trade(tradeParameters, buyLimit)
# Test performance of optimization.
#successRate = evaluate(results, buyLimit)
#print(f"Success Rate: {successRate}")

#print(trade([1,20,2], buyLimit))
# Report results.
#print("Final value: $" & results)
#if successRate > 0:
    #print("Trading bot performed " & successRate & "\% better than the __chosen baseline__.")
#elif successRate < 0:
    #print("Trading bot performed " & abs(successRate) & "% poorer than the __chosen baseline__. Output was not successfully optimized.")
#else:
    #print("Trading bot's performance matched the __chosen baseline__. Output was not successfully optimized.")


# FORWARD TESTING

# Only forward test if data is split
if split_ratio != 1:
    df = df_test # Reassign df to test dataframe

# Test against etherium data
else:
    eth_data = exchange.fetch_ohlcv('ETH/AUD', timeframe='1d', limit=720)
    df = pd.DataFrame(eth_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


# Indicator initialisation code copied from above (maybe put in a function)
bb_indicator = BollingerBands(df['close'],window=5)

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

#Simplifies the timestamp to be easier to use with triggers
df['simplified_timestamp'] = pd.to_numeric((df['timestamp'] - df['timestamp'].loc[0])  / (df['timestamp'].loc[2] - df['timestamp'].loc[1]), downcast='signed')


# Run trade with optimized parameters
#results = trade(tradeParameters)
#print("Test data results: ", results)

# Evalaute trade success
#successRate = evaluate(results,buyLimit)
#print("Test data success rate: ", successRate)
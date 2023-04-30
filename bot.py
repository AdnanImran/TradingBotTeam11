import ccxt
import ta
import pandas as pd
import random
#Bollinger Bands help identify sharp, short-term price movements and potential entry and exit points.
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator

#Sets up data & variables
exchange = ccxt.kraken()
bitcoin_data = exchange.fetch_ohlcv('BTC/AUD', timeframe='1d', limit=720)
df = pd.DataFrame(bitcoin_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
bb_indicator = BollingerBands(df['close'],window=5)
cash= 100.0
crypto=0.0

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
    best_solution = [] #dummy value until I know what to use
    best_profit= 0
    candidate_solution = [] #dummy value until I know what to use

    #Typical Parameters for our DE
    #D – Problem dimension -> Defined by the number of decision variables or parameters that are being optimized in the fitness function. 2 (or 3 - will we include fillna?)
    popSize = 10 #N – Population size (pop_var) -> How many input variables will be consider each time?
    recombinationValue = 0.4
    mutationValue = 0.6#F – Scaling factor ->  Controls the amplification of the difference vector (difference between two randomly selected individuals from the population) used in the mutation step.
    gen = 10  #G – Number of generation/stopping condition -> Decide how many iterations should be considered.     
    bounds = [(1,720),(0,100)] #Li,Hi – boundary for dimension i -> These boundaries help to constrain the search space of the algorithm.

    # Initialise Population and randomly pick values for the parameters
    population = initPopulation(popSize, bounds)
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

            '''Recombination section needs additional work to factor in both parameters'''

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

            newScore  = trade(selected)
            oldScore = trade(xT)

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
        gen_sol = population[gen_scores.index(min(gen_scores))]

        print('> GENERATION AVERAGE:',gen_avg)
        print('> GENERATION BEST:',gen_best)
        print('> BEST SOLUTION:',gen_sol,'\n')
    print(gen_sol)
    return gen_sol

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

# Initialise the population
# popSize --> Number of individuals in population
# bounds --> Bounds of each parameter 
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

# Tests the performance of the optimization algorithm.
def evaluate(df, results):
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

def trade(df, parameters):
    #buy and sell functions here?
    return None

'''Three Key Functions'''
# Generate optimal parameters.
tradeParameters = optimize()
# Run the trade.
results = trade(df, tradeParameters)
# Test performance of optimization.
effectiveness = evaluate(df, results)

# Report results.
print("Final value: $" & results)
if effectiveness > 0:
    print("Trading bot performed " & effectiveness & "\% better than the __chosen baseline__.")
elif effectiveness < 0:
    print("Trading bot performed " & abs(effectiveness) & "% poorer than the __chosen baseline__. Output was not successfully optimized.")
else:
    print("Trading bot's performance matched the __chosen baseline__. Output was not successfully optimized.")
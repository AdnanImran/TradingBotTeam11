import numpy as np
import random

# This is the function we are applying the optimisation on. Trying to get the best parameters
# that will minimise the value
# Here we will be trying to maximise the area of a circle
def targetFunction(x):
   return 4 * np.pi * np.power(x,2)

# This is the actual optimization algorithm
def optimise():
    # Setting the bounds for the parameters
    bounds = [(0,10)]
    # Population size
    popSize = 5
    # Number of iterations or generations
    maxIterations = 10
    # Mutation values and recombination/selection value
    mutationValue = 0.7
    # The higher the value, the more likely we'll take a mutated child as a solution, 
    # i.e. it wont get stuck on one solution
    recombinationValue = 0.4

    # Initialise population and randomly pick values for the parameters
    population = []
    for i in range(0,popSize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0],bounds[j][1]))
        population.append(indv)
    

    # Go through each generation
    for i in range(1,maxIterations+1):
        print('GENERATION:',i)

        # Keep track of each generations score
        gen_scores = [] 
        # Go through each individual in the population
        for j in range(0, popSize):

            #--- MUTATION -----
            # Mutation expands the search space
            # Select three random vector index positions [0, popSize), not including current vector (j)
            # from the current generation that are both unique to themselves, but also unique to the 
            # currently selected individual that we're mutating (j)
            candidates = list(range(popSize))
            candidates.remove(j)
            # This random.sample method will handle the unique selection
            # Parameters are the population and number of index positions to select
            random_index = random.sample(candidates, 3)

            # Get the values from the actual population
            x1 = population[random_index[0]]
            x2 = population[random_index[1]]
            x3 = population[random_index[2]]
            xT = population[j]

            # NOTE: Mutation strategy taken from this article: 
            # "Quantitative Trading Machine Learning Using Differential Evolution Algorithm"
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
            
            # Ensure that new values obtained through mutation are within the bounds of the parameters
            vectorDonor = checkBounds(vectorDonor, bounds)

            #--- RECOMBINATION/CROSSOVER ----------
            # Similar process to the article
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

            newScore  = targetFunction(selected)
            oldScore = targetFunction(xT)

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

# Ensuure that when 
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

optimise()
# PomAndAussies [AIGroup11]



## Setup
1.  Install required libraries.
```
pip3 install -r requirements.txt
```
2.  Lorem ipsum dolor sit amet, consectet

## Research
Some articles that might be useful:
- "An Automated Trading System with Multi-indicator Fusion Based on D-S Evidence Theory in Forex Market" by Liu, Zhihong ; Xiao, Deyun (2009)
- "Ensemble learning of rule-based evolutionary algorithm using multi layer perceptron for stock trading models" by Mabu, Shingo ; Obayashi, Masanao ; Kuremoto, Takashi (2014)

## Potential Algorithms

### 1. Genetic Algorithm

General framework of the algorithm
1. Randomly initialise a population of values/parameters we want to optimise. Each individual in the population represents a solution that we want to use.
2. while the stopping criterion is not met (E.g. Max number of generations) do:
    - evaluate how each individual in the population performs by calculating its fitness value
    - select the best performing individuals based on the fitness value to act as parents for the next generation
    - create children from the parents by selected parts that the parents have
    - randomly mutate some values from the children to create diversity and prevent algorithm from getting stuck
    - replace previous generation with the new generation
    - repeat while loop

#### 1.1 Genetic Pros & Cons
**Pros:**
- Similar to algorithms our team member wrote their essay on.
- Easy to understand and implement.

**Cons:**
- There's no garauntee that the random heuristics will find the optimum solution.
- Can get stuck at a local optimum point.

#### 1.2 Genetic Further Reading
- "An Intelligent Model for Pairs Trading Using Genetic Algorithms" by Chien-Feng Huang,1 Chi-Jen Hsu,1 Chi-Chung Chen,2 Bao Rong Chang,1 and Chen-An Li (2015) --> *Very similar to approach we're taking, and describes success of genetic algorithms for this type of problem.*
- "Incorporating Markov decision process on genetic algorithms to formulate trading strategies for stock markets" --> *Explains how this approach is optimal for helping investors solve timing problems.*

### 2. Swarm Algorithm (Specifically Particle Swarm Optimisation (PSO))
General framework of the algorithm
1. Randomly initialise a population of values/parameters we want to optimise. Each individual in the population represents a solution that we want to use.
2. while stopping criterion is not met do: 
    - evaluate how each individual performs by calculating its fitness value against a fitness function
    - determine the best performing agent as save that agents values/parameters as the global best
    - determine each agents individual best performance based on previous performances
    - update each agents position/weight based on its current performance, the global best performance and its own personal best performance (The way this update is done varies between different type of swarm algorithms)
    - repeat while loop
3. Return global best agent (agent that has the best parameter/values)

#### 2.0.1 Particle Swarm Optimisation Algorithm

The general framework of this algorithm follows typical swarm algorithms as described above, with the difference being how each agents position/weight is updated.

More specifically updating the velocity and position of each particle based on its current position, personal best position, and global best position. This is calculated using a formula that balances exploration (finding new solutions or unexplored areas in the solution space) and exploitation (moving towards optimizing the best-known solutions) 

#### 2.1 Swarm Pros & Cons
**Pros:**
- Generally faster than GA
- Easy to implement.
- Remembers the best solution found so far (helps with getting stuck at a local optimum point)

**Cons:**
- Can become trapped in local optima (due to it remembering the best solution)
- No garauntee that it will find the global optimum
- Based on the success reported in academic articles it may not be a good fit for this type of problem.

#### 2.2 Swarm Further Reading
- "A constrained portfolio trading system using particle swarm algorithm and recurrent reinforcement learning" by Saud Almahdi, Steve Y. Yang (2019) --> *Notes that while this algorithm optimized short term portfolios, it was not so effective at optimizing long term portfolios.*

- A. C. Briza and P. C. Naval, “Stock trading system based on the multi-objective particle swarm optimization of technical indicators on end-of-day market data,” Applied soft computing, vol. 11, no. 1, pp. 1191–1201, 2011, doi: 10.1016/j.asoc.2010.02.017. --> *Similar to the approach we will be taking*

### 3. Ant Colony Optimization (ACO)
General framework of the algorithm
1. 

#### 3.1 ACO Pros & Cons
**Pros:**
- Capable of solving complex optimization problems [*But is this project a complex problem? I would say no - Georgia*]

**Cons:**
- Easily falls into trap of local optimum
- Can be slow when dealing with large problem spaces. 

#### 3.2 ACO Further Reading
- 

### 4. Differential Evolution (DE)
General framework of the algorithm
1. Randomly initialize a population of candidate solutions.
2. While the stopping criterion is not met (E.g. Max number of iterations) do:
    - Generate new candidate solutions by combining existing solutions using a mutation operator. The mutation operator generates a new candidate solution By adding the difference between two other candidate solution to a third candidate solution.
    - Combine the original candidate solutions with the newly generated solutions using a crossover operator. The crossover operator selects components (e.g., variables or parameters) from each candidate solution to create a new solution.
    - Select the best candidate solutions from the combined population to create a new population for the next iteration. The selection process can be based on various criteria, such as fitness or objective function values.

#### 4.1 Differential Pros & Cons
**Pros:**
- Robust - Good for solution spaces with multiple local optima
- Is fastest algorithm of the 5 considered
- Can be parallelized to improve speed further [*We could consider this if we have extra time to improve the project - Georgia*]
- Beginner friendly [*Favourable as 40% of the group are new to AI - Georgia*]

**Cons:**
- No-one in the group wrote about this type of algorithm in assignment 1.
- Highly dependent on the control parameters involved [*As our parameters are fairly self-contained, I don't think this is an issue - Georgia*]


#### 4.2  Differential Further Reading
-  "Differential Evolution: A Survey and Analysis" by Tarik Eltaeib and Ausif Mahmood (2018) Access: https://www.mdpi.com/2076-3417/8/10/1945
- "Quantitative Trading Machine Learning Using Differential Evolution Algorithm" by Napas Vinitnantharat; Narit Inchan; Thatthai Sakkumjorn; Kitsada Doungjitjaroen; Chukiat Worasucheep (2019) Access: https://ieeexplore.ieee.org/document/8864226 --> *DE used to optimize trading strategy, **contains some pseudocode** and results. Might be good reference for the report and implementation. Their conclusion was this method worked well when market was fluctuating or trending downward, but performed poorly when market was trending upward.*

### 5. Artificial Bee Colony (ABC)
General framework of the algorithm
1. Initialize the population of food sources. Each food source represents a potential solution to the problem being optimized.
2. While the stopping criterion is not met (E.g. Max number of iterations) do:
    - Employed Bees Phase: In this phase, employed bees generate a new food source by selecting a random partner and modifying their solution by adding or subtracting a random value. If the modified solution is better than the original, the employed bee returns to the original position, otherwise, it continues to search for a better solution.
    - Onlooker Bees Phase: Onlooker bees observe the food sources found by employed bees and select a food source based on its fitness value. The probability of selecting a particular food source is proportional to its fitness value.
    - Scout Bees Phase: If an employed bee or onlooker bee cannot find a better solution after a certain number of iterations, a scout bee is generated. The scout bee randomly searches for a new solution.
    - Abandonment of Food Sources: Food sources with poor fitness values are abandoned by the employed bees and replaced by new solutions generated by the scout bees.

#### 5.1 ABC Pros & Cons
**Pros:**
- Is a global optimization algorithm - it is capable of finding the global optimimum in the search space
- Flexible - it can be applied to many different types of problems
- Efficient - can handle large-scale optimization problems
- The bee analogy is pretty cute...

**Cons:**
- Can have high memory usage when dealing with large datasets
- In some cases, the algorithm may focus too much on exploring a limited region of the search space, which can lead to suboptimal solutions.
- Sensitive to initialization: The ABC algorithm can be sensitive to the initial population of candidate solutions, which can lead to variability in the results obtained.

#### 5.2 ABC Further Reading
- 

## Measuring Success
"Modeling, forecasting and trading the EUR exchange rates with hybrid rolling genetic algorithms—Support vector regression forecast combinations" by Georgios Sermpinis, Charalampos Stasinakis, Konstantinos Theofilatos, Andreas Karathanasopoulos (2015) notes that  "statistical accuracy is not always synonymous with financial profitability." We as a group need to make a conscious decision about whether to hold the bot's performance against its statistical accuracy or its financial profitablility. Given the specifications of the assignment, we feel that the financial profitablility is the best measure of success. We are provided clear instructions to consider the value of each trade and if we measured success based on statistical accuracy exclusively we could discard this variable. 
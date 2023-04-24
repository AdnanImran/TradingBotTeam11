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

### 1.2 Genetic Pros & Cons
**Pros:**
    - Similar to one of the algorithms our team member wrote their essay on.
    - Easy to understand and implement.
    - Algorithm performance is quite fast when compared with other algorithms

**Cons:**
    - There's no garauntee that the random heuristics will find the optimum solution.
    - Can get stuck at a local maxima point.

### 1.3 Genetic Further Reading
    - "An Intelligent Model for Pairs Trading Using Genetic Algorithms" by Chien-Feng Huang,1 Chi-Jen Hsu,1 Chi-Chung Chen,2 Bao Rong Chang,1 and Chen-An Li (2015) --> Very similar to approach we're taking, and describes success of genetic algorithms for this type of problem.
    - "Incorporating Markov decision process on genetic algorithms to formulate trading strategies for stock markets" --> explains how this approach is optimal for helping investors solve timing problems.

### 2. Swarm Algorithm


### 2.2 Genetic Pros & Cons
**Pros:**
    - 

**Cons:**
    - 

### 2.3 Genetic Further Reading
    - 

## Measuring Success
    "Modeling, forecasting and trading the EUR exchange rates with hybrid rolling genetic algorithms—Support vector regression forecast combinations" by Georgios Sermpinis, Charalampos Stasinakis, Konstantinos Theofilatos, Andreas Karathanasopoulos (2015) notes that  "statistical accuracy is not always synonymous with financial profitability." We as a group need to make a conscious decision about whether to hold the bot's performance against its statistical accuracy or its financial profitablility. Given the specifications of the assignment, we feel that the financial profitablility is the best measure of success. We are provided clear instructions to consider the value of each trade and if we measured success based on statistical accuracy exclusively we could discard this variable. 
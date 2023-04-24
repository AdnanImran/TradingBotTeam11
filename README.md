# PomAndAussies [AIGroup11]



## Setup
1.  Install required libraries.
```
pip3 install -r requirements.txt
```
2.  Lorem ipsum dolor sit amet, consectet

## Research
Some articles that might be useful:
- "An intelligent hybrid trading system for discovering trading rules for the futures market using rough sets and genetic algorithms" by JKim, Youngmin ; Ahn, Wonbin ; Oh, Kyong Joo ; Enke, David (2017)
- "Developing an enhanced portfolio trading system using K-means and genetic algorithms" by Ahn, Wonbin ; Cheong, Donghyun ; Kim, Youngmin ; Oh, Kyong Joo (2018)
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

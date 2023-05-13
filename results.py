import bot
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import tabulate


def getResults(buyLimit, hyperparams): #hyperparams (popSize, recombinationValue, mutationValue, gen)
    return bot.optimize(buyLimit, hyperparams)

def plotResults(results,hyperparams):
    name = HypertoString(hyperparams)
    plt.plot(range(len(results[1])),results[1])
    plt.plot(range(len(results[2])),results[2])
    plt.plot(range(len(results[1])), [bot.trade([1,20,2,2], hyperparams[0])]*len(results[1]), linestyle='dotted')
    plt.legend(("Best Solution", "Average Solution", "Evaluation Target"))
    plt.figtext(0.8,0.8,"{:.2f}".format(results[1][-1]))
    plt.xlabel("Generations")
    plt.ylabel("Final Money")
    plt.title = name
    plt.savefig(os.path.join("graphs", name))
    plt.clf()
    
def HypertoString(hyperparams):
    s = str(hyperparams[0]) +","+ str(hyperparams[1]) + ","+ str(hyperparams[2]) +","+ str(hyperparams[3]) +","+ str(hyperparams[4])
    s=s.replace(".", "-")
    return s

def hyperList(finals, param):
    values = {}
    if param =="buyLimit":
        for x in buyLimits:
            values[x] = []
    if param =="recombinationValue":
        for x in recombinationValues:
            values[x] = []
    if param =="mutationValue":
        for x in mutationValues:
            values[x] = []
    for key in finals:
        val = finals[key][1][-1]
        buyLimit, popSize, recombinationValue, mutationValue, gen = key.split(",")
        if param == "buyLimit":
            values[int(buyLimit)].append(val)
        if param == "recombinationValue":
            values[float(recombinationValue.replace("-","."))].append(val)
        if param == "mutationValue":
            values[float(mutationValue.replace("-","."))].append(val)
        
    return values
            
def evalScores(res):
    for key in res:
        val = bot.evaluate(res[key][1][-1], int(key.split(",")[0]))
        eva.append(val)
    return eva

def loadResults(filename='all_final_results.txt'):
    f = open(filename)
    r = f.read()
    r = r.replace('\'', '"')
    r = r.replace('(', '[').replace(')',']')
    d = json.loads(r)
    return d

buyLimits = [10,30,90,150,360,720]
popSize = 5
recombinationValues = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
mutationValues = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
gen = 10

def getMeans(res, param, function, func_name):
    h = hyperList(res, param)
    table = [[param, func_name]]
    for key in h:
        table.append([key, function(h[key])])
    print(tabulate.tabulate(table))
    

counter = 1
finals = {}
for b in buyLimits:
    for r in recombinationValues:
        for m in mutationValues:
            res = getResults(b, (popSize, r, m, gen))
            plotResults(res, (b, popSize, r, m, gen))
            print(str(counter) + " out of " + str(len(buyLimits) * len(recombinationValues) * len(mutationValues)))
            finals[HypertoString((b, popSize, r, m, gen))] = res
            counter+=1

import bot
import matplotlib.pyplot as plt
import os

def getResults(buyLimit, hyperparams): #hyperparams (popSize, recombinationValue, mutationValue, gen)
    return bot.optimize(buyLimit, hyperparams)

def plotResults(results,hyperparams):
    name = HypertoString(hyperparams)
    plt.plot(range(len(results[1])),results[1])
    plt.plot(range(len(results[2])),results[2])
    plt.legend(("Best Solution", "Average Solution"))
    plt.figtext(0.8,0.8,"{:.2f}".format(results[1][-1]))
    plt.xlabel("Generations")
    plt.ylabel("Final Money")
    plt.title = name
    plt.savefig(os.path.join("graphs", name))
    
def HypertoString(hyperparams):
    s = str(hyperparams[0]) +","+ str(hyperparams[1]) + ","+ str(hyperparams[2]) +","+ str(hyperparams[3]) +","+ str(hyperparams[4])
    s=s.replace(".", "-")
    return s

buyLimits = [10,30,90,150,360,720]
popSize = 10
recombinationValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
mutationValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
gen = 10

counter = 1
for b in buyLimits:
    for r in recombinationValues:
        for m in mutationValues:
            res = getResults(b, (popSize, r, m, gen))
            plotResults(res, (b, popSize, r, m, gen))
            print(str(counter) + " out of " + str(len(buyLimits) * len(recombinationValues) * len(mutationValues)))
            counter+=1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GWSAT import GWSAT
from WalkSAT import WalkSAT
import pickle
import matplotlib.pyplot as plt
import statistics

fileName1 = r"C:\CIT MSc Repo\CIT MSc in AI\COMP9058\Assignment 2\uf20-01.cnf"
fileName2 = r"C:\CIT MSc Repo\CIT MSc in AI\COMP9058\Assignment 2\uf20-02.cnf"
files = ['uf20-01.cnf','uf20-02.cnf' ]
executions = 200
restarts = 10
iterations = 1000
wp = 0.4
tl = 5
iterationsToSolve = {}
GWSATOutputs = []

outputGWSAT1 = np.array(GWSAT(fileName1, executions, restarts, iterations, wp ).start()[0])
iterationsToSolve1 = outputGWSAT1[outputGWSAT1[:,0] == 1][:,2]
timeToSolve1 =  outputGWSAT1[outputGWSAT1[:,0] == 1][:,3] * 1000
print("done 1")
outputGWSAT2 = np.array(GWSAT(fileName2, executions, restarts, iterations, wp ).start()[0])
iterationsToSolve2 = outputGWSAT2[outputGWSAT2[:,0] == 1][:,2]
timeToSolve2=  outputGWSAT2[outputGWSAT2[:,0] == 1][:,3] * 1000
print("done 2")
outputWalkSAT1 = np.array(WalkSAT(fileName1, executions, restarts, iterations, wp, tl ).start()[0])
iterationsToSolve3 = outputWalkSAT1[outputWalkSAT1[:,0] == 1][:,2]
timeToSolve3 =  outputWalkSAT1[outputWalkSAT1[:,0] == 1][:,3] *1000
print("done 3")
outputWalkSAT2 = np.array(WalkSAT(fileName2, executions, restarts, iterations, wp, tl ).start()[0])
iterationsToSolve4 = outputWalkSAT2[outputWalkSAT2[:,0] == 1][:,2]
timeToSolve4 =  outputWalkSAT2[outputWalkSAT2[:,0] == 1][:,3] *1000
print("done 4")

dumpFile1 = "outputGWSAT1"
dumpFile2 = "outputGWSAT2"
dumpFile3 = "outputWalkSAT1"
dumpFile4 = "outputWalkSAT2"

fig, ax = plt.subplots()

ax.set_title('RTD - Search Steps')
ax.set_xlabel('Iterations')
ax.set_ylabel('P(Solve)')
y = np.arange(1, len(iterationsToSolve1)+1) / len(iterationsToSolve1)
ax.plot(sorted(iterationsToSolve1), y, linestyle='dotted', label='GWSAT uf20-01')
y2 = np.arange(1, len(iterationsToSolve2)+1) / len(iterationsToSolve2)
ax.plot(sorted(iterationsToSolve2), y2, linestyle='dotted', label='GWSAT uf20-02')
y3 = np.arange(1, len(iterationsToSolve3)+1) / len(iterationsToSolve3)
ax.plot(sorted(iterationsToSolve3), y3, linestyle='dotted', label='WalkSAT uf20-01')
y4 = np.arange(1, len(iterationsToSolve4)+1) / len(iterationsToSolve4)
ax.plot(sorted(iterationsToSolve4), y4, linestyle='dotted', label='WalkSAT uf20-02')
ax.legend(loc='lower right', fontsize=8)
plt.show()

fig, ax = plt.subplots()

ax.set_title('RTD - Search Steps (loglog)')
ax.set_xlabel('Iterations')
ax.set_ylabel('P(Solve)')
y = np.arange(1, len(iterationsToSolve1)+1) / len(iterationsToSolve1)
ax.loglog(sorted(iterationsToSolve1), y, linestyle='dotted', label='GWSAT uf20-01')
y2 = np.arange(1, len(iterationsToSolve2)+1) / len(iterationsToSolve2)
ax.loglog(sorted(iterationsToSolve2), y2, linestyle='dotted', label='GWSAT uf20-02')
y3 = np.arange(1, len(iterationsToSolve3)+1) / len(iterationsToSolve3)
ax.loglog(sorted(iterationsToSolve3), y3, linestyle='dotted', label='WalkSAT uf20-01')
y4 = np.arange(1, len(iterationsToSolve4)+1) / len(iterationsToSolve4)
ax.loglog(sorted(iterationsToSolve4), y4, linestyle='dotted', label='WalkSAT uf20-02')
ax.legend(loc='lower right', fontsize=8)
plt.show()

fig, ax = plt.subplots()

ax.set_title('RTD - Search Steps (semi-log)')
ax.set_xlabel('Iterations')
ax.set_ylabel('P(Solve)')
y = np.arange(1, len(iterationsToSolve1)+1) / len(iterationsToSolve1)
ax.semilogx(sorted(iterationsToSolve1), y, linestyle='dotted', label='GWSAT uf20-01')
y2 = np.arange(1, len(iterationsToSolve2)+1) / len(iterationsToSolve2)
ax.semilogx(sorted(iterationsToSolve2), y2, linestyle='dotted', label='GWSAT uf20-02')
y3 = np.arange(1, len(iterationsToSolve3)+1) / len(iterationsToSolve3)
ax.semilogx(sorted(iterationsToSolve3), y3, linestyle='dotted', label='WalkSAT uf20-01')
y4 = np.arange(1, len(iterationsToSolve4)+1) / len(iterationsToSolve4)
ax.semilogx(sorted(iterationsToSolve4), y4, linestyle='dotted', label='WalkSAT uf20-02')
ax.legend(loc='lower right', fontsize=8)
plt.show()


#Times


fig, ax = plt.subplots()

ax.set_title('RTD - Run Time')
ax.set_xlabel('Run time in msec')
ax.set_ylabel('P(Solve)')
y = np.arange(1, len(timeToSolve1)+1) / len(timeToSolve1)
ax.plot(sorted(timeToSolve1), y, linestyle='dotted', label='GWSAT uf20-01')
y2 = np.arange(1, len(timeToSolve2)+1) / len(timeToSolve2)
ax.plot(sorted(timeToSolve2), y2, linestyle='dotted', label='GWSAT uf20-02')
y3 = np.arange(1, len(timeToSolve3)+1) / len(timeToSolve3)
ax.plot(sorted(timeToSolve3), y3, linestyle='dotted', label='WalkSAT uf20-01')
y4 = np.arange(1, len(timeToSolve4)+1) / len(timeToSolve4)
ax.plot(sorted(timeToSolve4), y4, linestyle='dotted', label='WalkSAT uf20-02')
ax.legend(loc='lower right', fontsize=8)
plt.show()

fig, ax = plt.subplots()

ax.set_title('RTD - Run Time (loglog)')
ax.set_xlabel('Run time in msec')
ax.set_ylabel('P(Solve)')
y = np.arange(1, len(timeToSolve1)+1) / len(timeToSolve1)
ax.loglog(sorted(timeToSolve1), y, linestyle='dotted', label='GWSAT uf20-01')
y2 = np.arange(1, len(timeToSolve2)+1) / len(timeToSolve2)
ax.loglog(sorted(timeToSolve2), y2, linestyle='dotted', label='GWSAT uf20-02')
y3 = np.arange(1, len(timeToSolve3)+1) / len(timeToSolve3)
ax.loglog(sorted(timeToSolve3), y3, linestyle='dotted', label='WalkSAT uf20-01')
y4 = np.arange(1, len(timeToSolve4)+1) / len(timeToSolve4)
ax.loglog(sorted(timeToSolve4), y4, linestyle='dotted', label='WalkSAT uf20-02')
ax.legend(loc='lower right', fontsize=8)
plt.show()

fig, ax = plt.subplots()

ax.set_title('RTD - Run Time (Semi log)')
ax.set_xlabel('Run time in msec')
ax.set_ylabel('P(Solve)')
y = np.arange(1, len(timeToSolve1)+1) / len(timeToSolve1)
ax.semilogx(sorted(timeToSolve1), y, linestyle='dotted', label='GWSAT uf20-01')
y2 = np.arange(1, len(timeToSolve2)+1) / len(timeToSolve2)
ax.semilogx(sorted(timeToSolve2), y2, linestyle='dotted', label='GWSAT uf20-02')
y3 = np.arange(1, len(timeToSolve3)+1) / len(timeToSolve3)
ax.semilogx(sorted(timeToSolve3), y3, linestyle='dotted', label='WalkSAT uf20-01')
y4 = np.arange(1, len(timeToSolve4)+1) / len(timeToSolve4)
ax.semilogx(sorted(timeToSolve4), y4, linestyle='dotted', label='WalkSAT uf20-02')
ax.legend(loc='lower right', fontsize=8)
plt.show()

def statsExec(stats):
        
    outcomes = stats[:,0]
    trueOutcomes = outcomes.sum()

    if trueOutcomes == 0:
        successRate = 0
    else:
        successRate =  trueOutcomes / (len(outcomes))
    runTimes = stats[:,3] * 1000
    averageRunTime = runTimes.mean()
    maxRuntime = np.amax(runTimes)
    minRuntime = np.amin(runTimes)
    iterations = stats[:,2]
    iterationsToSolve = stats[stats[:,0] == 1][:,2]

    
    print("Executions: {}".format(executions))
    print("Total iterations: {}". format(np.sum(iterations)))
    
    if len(iterationsToSolve) > 0 :
        print("Iterations to solve: \n Max: {0}\n Min: {1}\n Mean: {2}\n Median: {3}\n STDEV: {4:.2f}".format(np.amax(iterationsToSolve), np.amin(iterationsToSolve), 
            statistics.mean(iterationsToSolve), statistics.median(iterationsToSolve), statistics.stdev(iterationsToSolve)))
    print("Success rate: {0:.0%}".format(successRate))
    print("Run time per restart (ms): \n Max: {0:.3f}\n Min: {1:.3f}\n Mean: {2:.3f}\n Median: {3:.3f}\n STDEV: {4:.3f}".format(np.amax(runTimes), np.amin(runTimes), 
        runTimes.mean(),statistics.median(runTimes), runTimes.std()))
    print("Iterations: \n Max: {0}\n Min: {1}\n Mean: {2:.2f}\n Median: {3:.2f}\n STDEV: {4:.2f}".format(np.amax(iterations), np.amin(iterations), iterations.mean(), 
        statistics.median(iterations), iterations.std()))

print("GWSAT 1")
statsExec(outputGWSAT1)
print("GWSAT 2")
statsExec(outputGWSAT2)

print("Walk 1")
statsExec(outputWalkSAT1)
print("Walk 2")
statsExec(outputWalkSAT2)

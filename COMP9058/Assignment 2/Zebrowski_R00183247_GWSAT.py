from GWSAT import GWSAT
import numpy as np
import statistics
import sys



if len(sys.argv) < 6:
    print ("Error - Incorrect input")
    print ("Expecting python Zebrowski_R00183247_GWSAT.py instance executions restarts iterations wp")
    sys.exit(0)
    



instance = sys.argv[1]
executions = int(sys.argv[2])
restarts = int(sys.argv[3])
iterations = int(sys.argv[4])
wp = float(sys.argv[5])



#instance = r"C:\CIT MSc Repo\CIT MSc in AI\COMP9058\Assignment 2\uf20-01.cnf"
#executions = 30
#restarts = 10
#iterations = 1000
#wp = 0.7



print("Processing the following: \nInstance: {} \nExecutions: {} \nRestarts: {} \nIterations: {} \nwp: {} \n".format(
        instance, executions, restarts, iterations, wp))

#Initialise variables
c1 = GWSAT(instance, executions, restarts, iterations, wp )
#execute search
stats, actualRestarts, start, end = c1.start()

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

print("Total run time (s): {0:.3f}".format(end - start))
print("Executions: {}".format(executions))
print("Total iterations: {}". format(np.sum(iterations)))
print("Restarts executed: {}".format(actualRestarts))
if len(iterationsToSolve) > 0 :
    print("Iterations to solve: \n Max: {0}\n Min: {1}\n Mean: {2}\n Median: {3}\n STDEV: {4:.2f}".format(np.amax(iterationsToSolve), np.amin(iterationsToSolve), 
        statistics.mean(iterationsToSolve), statistics.median(iterationsToSolve), statistics.stdev(iterationsToSolve)))
print("Success rate: {0:.0%}".format(successRate))
print("Run time per restart (ms): \n Max: {0:.3f}\n Min: {1:.3f}\n Mean: {2:.3f}\n Median: {3:.3f}\n STDEV: {4:.3f}".format(np.amax(runTimes), np.amin(runTimes), 
    runTimes.mean(),statistics.median(runTimes), runTimes.std()))
print("Iterations: \n Max: {0}\n Min: {1}\n Mean: {2:.2f}\n Median: {3:.2f}\n STDEV: {4:.2f}".format(np.amax(iterations), np.amin(iterations), iterations.mean(), 
    statistics.median(iterations), iterations.std()))



"""
Author: Adam Zebrowski
file: TSP_R00183247.py
"""

import random
from Individual import Individual
import sys
import time
from tsp import NNHeuristic
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



#My student numner R00183247, assigning seed to 183247
myStudentNum = 183247 # Replace 12345 with your student number

random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _case = 1):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.case           = _case
        self.stats          = []
        self.elite          = []

        self.readInstance()
        self.initPopulation()
        if _case == 7 or _case == 8:
            #For cases 7 and 8 you need to apply heuristic to the initial population
            #the function below runs tsp neigherest neighbour to apply the heuristic
            self.applyHeuristic()


    def readInstance(self):
        """
        Reading an instance from fName
        """

        self.genSize = int(np.loadtxt(self.fName, delimiter=' ', max_rows=1))
        #load the data from file into the numpy array. 
        #change from the original implementation is needed tot falilitate the tsp heuristic
        self.dataArray = np.loadtxt(self.fName, delimiter=' ', dtype={'names': ('i', 'x', 'y'), 'formats': ('int', 'int64', 'int64')}, skiprows=1)
        self.data = {}
        for line in self.dataArray:
            self.data[line[0]] = (line[1], line[2])
       


    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())
        
    def applyHeuristic(self):
        """
        Applying NN heuristic to the population
        """

        nn = NNHeuristic()
        for ind in self.population:
            #apply NN heuristic to the population.
            #this will generate already potentially optimal solution
            genes = nn.processNN(self.genSize ,self.dataArray, self.data)
            #Testing with the example solution provided for Lab1
            #genes, __x__ = nn.insertion_heuristic2(self.data)
            ind.setGene(genes)
            ind.computeFitness()
            
        #another option below is to generate a new population instead of modifying the existing one
        '''
        for i in range(0, self.popSize):
            newData, genes = nn.processNN(self.genSize ,self.dataArray, self.data)
            individual = Individual(self.genSize, newData, genes)
            individual.computeFitness()
            self.population.append(individual)
        '''
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best heuristic sol: ",self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        F = sum(1/p.fitness for p in self.population)
        N = self.popSize

        uniformDistance = 1 / N
        startingPoint = random.uniform(0, uniformDistance)

        markers = [startingPoint + (i*uniformDistance) for i in range(N)]
        chromosomeRange = []
        position = 0.0
        for p in self.population:
            f = (1/p.fitness)/ F
            chromosomeRange.append((position, position + f, p)) 
            position += f
        
        #print(chromosomeRange)
        selectedParents = []
        for m in markers:
            for c in chromosomeRange:
                if  m > c[0] and m <= c[1]:
                    selectedParents.append(c[2])
        
        #this approach replacs whole population with SUS sampled one
        return selectedParents
        

        ''' This option caters for the possibility to sample 2 parents for each iteration
        indA = selectedParents[ random.randint(0, N - 1) ]
        indB = selectedParents[ random.randint(0, N - 1) ]
        
        return indA, indB
        '''

        

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        #t = time.process_time()
        probability = 0.5
        genesA = indA.genes
        genesB = indB.genes
        genesChildA = []
        genesChildB = []
        #numberOfGenes = len(genesA)
        for g in range(0, self.genSize):
            if random.random() > probability:
                genesChildA.append(genesA[g])
                genesChildB.append(genesB[g])
            else:
                #Adding dummy values that will be replaced later
                genesChildA.append(-1)
                genesChildB.append(-1)

        #generating genes differences to make sure I do not have duplicates
        toAddToA = set(genesB) - set(genesChildA)
        toAddToB = set(genesA) - set(genesChildB)
        #toAddToA = list(temp for temp in genesB if temp not in genesChildA)
        #toAddToB = list(temp for temp in genesA if temp not in genesChildB)
        
        #populate the placeholders with genes from other parent respecively
        for g in range(0, self.genSize):
            if genesChildA[g] == -1:
                genesChildA[g] = toAddToA.pop()
            if genesChildB[g] == -1:
                genesChildB[g] = toAddToB.pop()
        
        #generate 2 new offsprings
        child1 = Individual(indA.genSize, indA.data, genesChildA)
        child2 = Individual(indB.genSize, indB.data, genesChildB)
        child1.computeFitness()
        child2.computeFitness()
        #elapsedTime = time.process_time() - t
        #print("Uniform xOver time {0}.".format(elapsedTime))
        return child1, child2
        

    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """

        cutPoint = random.randint(0, self.genSize - 1)
        
        endPoint = random.randint(cutPoint, (self.genSize - 1))
        #get the chunk of genes that weill be swapped
        listOfPositions = range(cutPoint, endPoint )
        genesA = indA.genes
        genesB = indB.genes
        genesChildA = []
        genesChildB = []
        relationshipTableAB = {}
        relationshipTableBA = {}

        #generate relationship table
        for position in listOfPositions:
            relationshipTableAB[genesB[position]] = genesA[position]
            relationshipTableBA[genesA[position]] = genesB[position]

        for g in range(0, self.genSize):
            if g >= cutPoint and g < endPoint:
                #cross the selected genes
                genesChildA.append(genesB[g])
                genesChildB.append(genesA[g])
            else:
                #apply placeholder to the rest
                genesChildA.append(-1)
                genesChildB.append(-1)
        
        #poplate the genes from x individuals
        for g in range(0, self.genSize):
            if genesChildA[g] == -1:
                #make sure there are no dupicates
                if genesA[g] not in genesChildA:
                    genesChildA[g] = genesA[g]
                   
            if genesChildB[g] == -1:
                if genesB[g] not in genesChildB:
                    genesChildB[g] = genesB[g]
                    
        #TODO: See if 2 loops can be combined
        for g in  range(0, self.genSize):
            if genesChildA[g] == -1:
                key = genesA[g]
                found = False
                #Loop relationship table to find the missing ones
                while not found:
                    if relationshipTableAB[key] not in genesChildA:
                        genesChildA[g] = relationshipTableAB[key]
                        found = True
                    else:
                        key = relationshipTableAB[key]

        for g in  range(0, self.genSize):
            if genesChildB[g] == -1:
                key = genesB[g]
                found = False
                #Loop relationsship table to find the missing genes
                while not found:
                    if relationshipTableBA[key] not in genesChildB:
                        genesChildB[g] = relationshipTableBA[key]
                        found = True
                    else:
                        key = relationshipTableBA[key]

        child1 = Individual(indA.genSize, indA.data, genesChildA)
        child2 = Individual(indB.genSize, indB.data, genesChildB)
        #child1.genes = genesChildA
        #child2.genes = genesChildB
        child1.computeFitness()
        child2.computeFitness()
        return child1, child2

        
        
        
    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        
        toMutate = int((self.genSize * self.mutationRate)) * 2
        geneToSwap = random.sample(range(0, self.genSize - 1), toMutate)
        
        for i in range(0, toMutate - 2):
            ind.genes[geneToSwap[i]], ind.genes[geneToSwap[i+1]] = ind.genes[geneToSwap[i+1]], ind.genes[geneToSwap[i]] 

        ind.computeFitness()
        self.updateBest(ind)
               
        


    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        if random.random() > self.mutationRate:
            return
        #select 2 starting points and establish a range
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)
        mutationRange = [indexA, indexB]
        mutationRange.sort()

        genes = ind.genes
        #slice the original genes and reverse
        genesToReverse = genes[mutationRange[0]: mutationRange[1]]
        genesToReverse.reverse()
        #apply the reverse genes to the chromosome
        genes[mutationRange[0]: mutationRange[1]] = genesToReverse
        ind.genes = genes
        ind.computeFitness()
        self.updateBest(ind)



    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        if self.case > 2:
            #for cases 3-8 the SUS sampling from the population is applied as a base for the new generation
            source = self.stochasticUniversalSampling()
        else:
            source = self.population

        for ind_i in source:
            self.matingPool.append( ind_i.copy() )

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        newPopulation = []
        eliteCandidate = []
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
           
            if len(newPopulation) > self.popSize - 2:
                break
            indA, indB = self.randomSelection()
            if self.case == 1:
                
                c1, c2 = self.uniformCrossover(indA, indB)
                self.inversionMutation(c1)
                self.inversionMutation(c2)
            elif self.case == 2:
               
                c1, c2 = self.pmxCrossover(indA, indB)
                self.reciprocalExchangeMutation(c1)
                self.reciprocalExchangeMutation(c2)
            elif self.case == 3:
                #indA, indB = self.stochasticUniversalSampling()
                c1, c2 = self.uniformCrossover(indA, indB)
                self.reciprocalExchangeMutation(c1)
                self.reciprocalExchangeMutation(c2)
            elif self.case == 4 or self.case == 7:
                #indA, indB = self.stochasticUniversalSampling()
                c1, c2 = self.pmxCrossover(indA, indB)
                self.reciprocalExchangeMutation(c1)
                self.reciprocalExchangeMutation(c2)
            elif self.case == 5:
                #indA, indB = self.stochasticUniversalSampling()
                c1, c2 = self.pmxCrossover(indA, indB)
                self.inversionMutation(c1)
                self.inversionMutation(c2)
            elif self.case == 6 or self.case == 8:
                #indA, indB = self.stochasticUniversalSampling()
                c1, c2 = self.uniformCrossover(indA, indB)
                self.inversionMutation(c1)
                self.inversionMutation(c2)

            #childred survive only if fittern than parents
            '''
            if indA.getFitness() > c1.getFitness():
                newPopulation.append(c1)
            else:
                newPopulation.append(indA)
            
            if indB.getFitness() > c2.getFitness():
                newPopulation.append(c2)
            else:
                newPopulation.append(indB)
            '''
            
             
            #Ensuring elitism. Only fitter chromosomes are retained
            
            if indA.getFitness() > c1.getFitness():
                eliteCandidate.append([c1.getFitness(), c1])
            
            if indB.getFitness() > c2.getFitness():
                eliteCandidate.append([c2.getFitness(), c2])
            
            
            
            #Option for keepint only 1 offspring which is fitter. 
            ''' if c1.getFitness() < c2.getFitness():
                newPopulation.append(c1)
            else:
                newPopulation.append(c2)
            '''

            #Option for keeping both offspring regardless of performance
            
            newPopulation.append(indA)
            newPopulation.append(indB)
            
        
                
        self.population = newPopulation
        #ec = np.array(eliteCandidate)


    
        

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """
        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        #self.applyNNHeuristic()
        self.stats = []
        self.iteration = 0
        while self.iteration < self.maxIterations:
            #t = time.process_time()
            self.GAStep()
            #elapsedTime = time.process_time() - t
            #print("Iteration: {0} ran in {1}.".format(self.iteration, elapsedTime))
            self.stats.append([self.iteration, self.best.getFitness()])
            #print("Iteration {} fitness: {}".format(self.iteration))
            self.iteration += 1
        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness())
        return self.iteration, self.best.getFitness(), self.stats

if len(sys.argv) < 7:
    print ("Error - Incorrect input")
    print ("Expecting python TSP_R00183247.py instance[,instance] runs population[,population] mutationRate[,mutationRate] iterations cases")
    sys.exit(0)
    


#problem_file = sys.argv[1]
files = sys.argv[1].split(",")
cases = sys.argv[6].split(",")
mutationRates = sys.argv[4].split(",")
populations = sys.argv[3].split(",")
plot = False

#problem_file = "C:\\CIT MSc Repo\\CIT MSc in AI\\COMP9058\\Assignment 1\\TSP_Project\\inst-20.tsp"
#files = []
#files.append(problem_file)
parameters = {"NRuns": int(sys.argv[2]), "Population": populations , "MutationRate": mutationRates, 
    "Iterations": int(sys.argv[5]), "Cases": cases, "Files": files}
#parameters = {"NRuns": sys.argv[2], "Population": 50, "MutationRate": 0.1, "Iterations": 100, "Case": 1}

def ExecuteData():
    ga = BasicTSP(parameters["Files"][0], int(populations[0]), float(mutationRates[0]), parameters["Iterations"], int(cases[0]))
    iterationsExecuted, bestFitness, iterationFitness = ga.search()
    data = pd.DataFrame(iterationFitness, columns=["Iteration Number", "Fitness"] )
    data.to_csv("data.csv")

#ExecuteData()

def ExecuteAnalysis():
    startTime = datetime.now()
    print("Starting on: {}".format(startTime))
    for file in parameters["Files"]:
        print("Starting on file: {}".format(file))
        with open("report_" + file, 'w') as f:
            f.write("\nStart report: file {} on: {} \n".format(file, startTime))
            f.flush()
            for case in parameters["Cases"]:
                print("Starting on case: {}".format(case))
                f.write("Case: {} \n\n".format(case))
                for population in populations:
                    print("Starting on population: {}".format(population))
                    f.write("Population number: {} \n\n".format(population))
                    for mutationRate in mutationRates:
                        print("Starting on mutation: {}".format(mutationRate))
                        f.write("Mutation rate: {} \n\n".format(mutationRate))
                        outputReport = []
                        reportItems = []
                        stats = []
                        for i in range(0, parameters["NRuns"]):
                            t = time.process_time()
                            ga = BasicTSP(file, int(population), float(mutationRate), parameters["Iterations"], int(case))
                            
                            iterationsExecuted, bestFitness, iterationFitness = ga.search()

                            elapsedTime = time.process_time() - t
                            stats.append([i, bestFitness])
                            reportItems.append([int(i), str(round(bestFitness, 2)),  round(elapsedTime, 2)])                    
                        values = np.array(stats)
                        data = pd.DataFrame(reportItems, columns=["Run Number", "Best Fitness", "Run Time"] )
                        reportHeader = "\n\nStatistics for file: {}, \nPopulation: {}, \nMutation rate: {}, \nIterations: {}, \nCase: {}, \nRuns: {} \n\n".format(file, 
                            population, mutationRate, parameters["Iterations"], case, parameters["NRuns"])
                        reportDetail =  data.__repr__()      
                        bf = values[:,1]
                        reportFooter = "\n\nRun over {} iterations, file: {}. \nBest Performance: {}, \nMean perfomrance: {}, \nWorst performance: {} \n".format(parameters["NRuns"], 
                            file, np.min(values[:,1]), np.average(values[:,1]), np.max(values[:,1]))
                        print(reportHeader)
                        print(reportDetail)
                        print(reportFooter)
                        f.write(reportHeader)
                        f.write(reportDetail)
                        f.write(reportFooter)
                        f.write("\n\n")
                        f.flush()
            #pd.DataFrame(reportItems).to_csv('aa.csv')
    endTIme = datetime.now()
    print("Finished on: {}, duration: {}".format(endTIme, endTIme - startTime))

ExecuteAnalysis()
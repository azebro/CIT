

"""
Author: Adam Zebrowski - Case 2
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys
import time

#My student numner R00183247, assigning seed to 183247
myStudentNum = 183247 # Replace 12345 with your student number

random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
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

        self.readInstance()
        self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

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
        F = sum(p.fitness for p in self.matingPool)

        #print(F)
        #I'm going to select 50% population
        #N = int(self.popSize * 0.5)
        N = 2
        P = F / N
        startingPoint = random.uniform(0, P)
        markers = [startingPoint + (i * P) for i in range(N)]
        chromosomeRange = []
        position = 0.0
        for p in self.matingPool:
            chromosomeRange.append((position, position + p.fitness, p)) 
            position += p.fitness
        
        #print(chromosomeRange)
        selectedParents = []
        #Parents may be selected multiple times, as the array is not filtered to single instance
        for m in markers:
            for c in chromosomeRange:
                if m > c[0] and m <= c[1]:
                    selectedParents.append(c[2])
        
        #indA = selectedParents[ random.randint(0, N - 1) ]
        #indB = selectedParents[ random.randint(0, N - 1) ]
        
        return selectedParents[0], selectedParents[1]

        

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
                genesChildA.append(-1)
                genesChildB.append(-1)

        toAddToA = set(genesB) - set(genesChildA)
        toAddToB = set(genesA) - set(genesChildB)
        #toAddToA = list(temp for temp in genesB if temp not in genesChildA)
        #toAddToB = list(temp for temp in genesA if temp not in genesChildB)
        
        for g in range(0, self.genSize):
            if genesChildA[g] == -1:
                genesChildA[g] = toAddToA.pop()
               
        #for g in range(0, numberOfGenes):
            if genesChildB[g] == -1:
                genesChildB[g] = toAddToB.pop()
                
                
                '''
                for gA in genesA:
                    if gA not in genesChildB:
                        genesChildB[g] = gA
                        break
                    '''
                '''
                for gB in (temp for temp in genesB if temp not in genesChildA):
                
                    if gB not in genesChildA:
                        genesChildA[g] = gB
                        break
                    '''
        
        child1 = Individual(indA.genSize, indA.data, genesChildA)
        child2 = Individual(indB.genSize, indB.data, genesChildB)
        #child1.genes = genesChildA
        #child2.genes = genesChildB
        child1.computeFitness()
        child2.computeFitness()
        #elapsedTime = time.process_time() - t
        #print("Uniform xOver time {0}.".format(elapsedTime))
        return child1, child2
        
      
        


    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """

        #probability = 50
        cutPoint = random.randint(0, self.genSize - 1)
        
        endPoint = random.randint(cutPoint, (self.genSize - 1))
        listOfPositions = range(cutPoint, endPoint )
        genesA = indA.genes
        genesB = indB.genes
        genesChildA = []
        genesChildB = []
        relationshipTableAB = {}
        relationshipTableBA = {}

        for position in listOfPositions:
            relationshipTableAB[genesB[position]] = genesA[position]
            relationshipTableBA[genesA[position]] = genesB[position]

        for g in range(0, len(genesA)):
            if g >= cutPoint and g < endPoint:
                genesChildA.append(genesB[g])
                genesChildB.append(genesA[g])
            else:
                genesChildA.append(-1)
                genesChildB.append(-1)
        
        for g in range(0, len(genesA)):
            if genesChildA[g] == -1:
                if genesA[g] not in genesChildA:
                    genesChildA[g] = genesA[g]
                   
            if genesChildB[g] == -1:
                if genesB[g] not in genesChildB:
                    genesChildB[g] = genesB[g]
                    

        for g in  range(0, len(genesA)):
            if genesChildA[g] == -1:
                key = genesA[g]
                found = False
                while not found:
                    if relationshipTableAB[key] not in genesChildA:
                        genesChildA[g] = relationshipTableAB[key]
                        found = True
                    else:
                        key = relationshipTableAB[key]

        for g in  range(0, len(genesB)):
            if genesChildB[g] == -1:
                key = genesB[g]
                found = False
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
        genes = ind.genes
        toMutate = int((self.genSize * self.mutationRate))
        geneToSwap = random.sample(range(0, self.genSize - toMutate - 1), 1)
        #geneToSwap.sort()
        #genesToReverse = genes[slice(geneToSwap[0], geneToSwap[1])]
        startPoint = geneToSwap[0]
        stopPoint = geneToSwap[0] + toMutate
        genesToReverse = genes[startPoint: stopPoint]
        genesToReverse.reverse()
        genes[startPoint: stopPoint] = genesToReverse
        
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
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        newPopulation = []
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            '''
            #Case 1
            if len(newPopulation) > self.popSize - 2:
                break
            indA, indB = self.randomSelection()
            c1, c2 = self.uniformCrossover(indA, indB)
            self.inversionMutation(c1)
            self.inversionMutation(c2)
            #if c1.getFitness() < c2.getFitness():
            newPopulation.append(c1)
            #else:
            newPopulation.append(c2)
            '''
            
            #Case 2
            '''
            if len(newPopulation) > self.popSize - 2:
                break
            indA, indB = self.randomSelection()
            c1, c2 = self.pmxCrossover(indA, indB)
            self.reciprocalExchangeMutation(c1)
            self.reciprocalExchangeMutation(c2)

            '''

            #Case 3
            if len(newPopulation) > self.popSize - 2:
                break

            #Case 3
            indA, indB = self.stochasticUniversalSampling()
            c1, c2 = self.pmxCrossover(indA, indB)
            self.reciprocalExchangeMutation(c1)
            self.reciprocalExchangeMutation(c2)

            newPopulation.append(c1)
            
            newPopulation.append(c2)


        
                
        self.population = newPopulation


        

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
        self.iteration = 0
        while self.iteration < self.maxIterations:
            t = time.process_time()
            self.GAStep()
            elapsedTime = time.process_time() - t
            print("Iteration: {0} ran in {1}.".format(self.iteration, elapsedTime))
            self.iteration += 1
            

        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness())
'''
if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)
    '''


#problem_file = sys.argv[1]

problem_file = "C:\\CIT MSc Repo\\CIT MSc in AI\\COMP9058\\Assignment 1\\TSP_Project\\inst-7.tsp"

ga = BasicTSP(problem_file, 100, 0.005, 500)

#ga.stochasticUniversalSampling()
#ga.uniformCrossover(ga.population[0], ga.population[1])
#ga.inversionMutation(ga.population[0])
#ga.pmxCrossover(ga.population[0], ga.population[1])
ga.search()
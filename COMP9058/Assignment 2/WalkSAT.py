import sys
import random
import timeit
import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt

class WalkSAT:
       
    
    '''
    Arguments:
        instance - the file to be processed
        executions - number or exections to run
        restarts - number of restarts
        iterations - number of iterations for attempted solution
        wp - the probability of executing 3a and 3b
        tp - length of tabu list
    '''    
    def __init__(self, instance, executions, restarts, iterations, wp, tl):
        self.insance = instance
        self.executions = executions
        self.restarts = restarts
        self.iterations = iterations
        self.wp = wp
        self.tabuList = [0] * tl
        self.tl = tl
        #the actual clauses processed from the source file
        self.clauses = []   
        #number of variables
        self.variablesCount = 0      
        #number of clauses
        self.numClause = 0   
        #truth statement
        self.truthStatement = []    
        #list of literals by clause    
        self.literals = {}   
        #list of all clauses not satisfied by trueth statement
        self.falseClauses = [] 
        #used to check for completeness, if > 0 -> clause is sat
        self.numSatisfiedLitsPerClause = {} 
        #variables included in false clauses
        self.falseVariables = [] 
        

    def start(self):
        start = timeit.default_timer()
        
        runStatistics = []
        iterationsToSolve = []
        
        self.readFile(self.insance)
        self.createContainingClauses()
        actualRestarts = 0
                 
        for e in range(0, self.executions):
            random.seed(183247 + e * 1000)

            for r in range(0, self.restarts):
                
                self.generateRandomTStatement()
                self.setFalseClauses()
                outcome = self.Solve(self.iterations)
                runStatistics.append(outcome)
                if outcome[0] == 1:
                    iterationsToSolve.append(r)
                    break
                actualRestarts += 1 

                        
           
        end = timeit.default_timer()
        stats = np.array(runStatistics)

        return stats, actualRestarts, start, end
        
        
    def Solve(self, iterations):
        solveStart = timeit.default_timer() 
        returnData = []
        flips = 0
        isSat = 0
        for iteration in range(1, iterations + 1):
            flips = iteration
            '''
            step 1 - choose random unsat clause BC
            extended for tabu: BC is filtered by the tabu list
            '''
            C = self.checkTabu(self.getClause())
            #if all variables selected are on tabu list, proceed for next iteration
            #this potentially could be modified to selecting a clause until one is 
            #found that has at least 1 variable not on tabu list
            if len(C) == 0:
                #iteration will be executed, hence 1 variable should pop out of the tabu list
                self.tabuList.pop(0)
                continue
            #check for step 2 from the assignment
            if not self.flipNotNegativeGain(C):
                r = random.random()
                if (r <= self.wp):
                    #step 3a - "Otherwise, with probability p, select random variable from BC to flip"
                    x = random.choice(C) - 1
                    self.flip(x)
                else:
                    #step 3b - with - "probability (1-p), select variable in BC with minimal negative gain (break ties randomly)""
                    self.pickPositionAndFlip(C)
            
            self.setFalseClauses()
            if  self.complete():
                isSat = 1
                break
                
         
        solveStop = timeit.default_timer()
        returnData = [isSat, self.truthStatement, flips, solveStop - solveStart, len(self.falseClauses)]
        return returnData
        
       
    def checkTabu(self, C):
        #convert literals to variables
        variables = [abs(v) for v in C]
        #subtract tabu list from the list of variables in C
        left = list(set(variables) - set(self.tabuList))
        return left

        

    '''
    Generate radom T statement, based on choice between positive and negative
    '''
    def generateRandomTStatement(self):
        self.truthStatement = []
        for x in range(1, self.variablesCount + 1):
            self.truthStatement.append(x * random.choice([-1, 1]))


    def readFile(self, file):
        defined = 0
        with open(file, 'r') as f:
            data = f.readlines()
        for line in data:
            words = line.split()
            if (words[0] is '%'):
                break
            if (words[0] is 'c'):
                continue
            elif(words[0] is not 'c' and defined is not 1):
                defined = 1
                self.variablesCount = int(words[2])
                self.numClause = int(words[3])
            else:
                temp = []
                for x in words:
                    y = int(x)
                    if y is not 0:
                        temp.append(int(x))
                self.clauses.append(temp)
        

    

    '''
    Split clauses per literals that are contained inside
    this will be used to asses the implication of variable flipping
    '''
    def createContainingClauses(self):
        for x in range(1, self.variablesCount + 1):
            #positives
            self.literals[x] = []
            #negatives
            self.literals[x * -1] = []
            for instance in self.clauses:
                if x in instance:
                    self.literals[x].append(instance)
                if (x * -1) in instance:
                    self.literals[x * -1].append(instance)

    #Get a random false clause
    def getClause(self):
        return random.choice(self.falseClauses)


    '''
    Build a list of unsat clauses and define # of satisfied 
    literarls per clause. THis will be used to evaluate flipped variables
    '''
    def setFalseClauses(self):
        self.falseClauses = []
        sats = 0
        v = []
        for instance in self.clauses:
            for x in self.truthStatement:
                if x in instance:
                    sats += 1
            #if no satisfied exists, add to unsat
            if sats is 0:
                self.falseClauses.append(instance)
                v += instance
            #capture number of literals that satisfy the clause
            #this will be used for net gain calculation
            self.numSatisfiedLitsPerClause[str(instance)] = sats  
            sats = 0
        #capture variables present in unsat clauses
        #this ssumes that I will count each variable once
        #but another approach would be to count those based on actual occurance in the clauses
        #that would increase probability of picking the variable involved in most unsat clauses
        self.falseVariables = list(set([abs(l) for l in v]))

      
    def complete(self):
        return len(self.falseClauses) == 0

    
    def flip(self,x):
        #self.flips +=1
        self.truthStatement[x] = self.truthStatement[x] * -1
        if len(self.tabuList) == self.tl:
            self.tabuList.pop(0)
        self.tabuList.append(x + 1)
        
    
    def flipNotNegativeGain(self, C):
        ties = []
        gainFound = False
        for x in C:
            #gained sat clauses as a result of var flipping
            gain = 0
            #clauses becoming unsat as a result for the flip
            loss = 0

            #As suggested during lecture, will only look at the clauses that 
            #contain the variable

            '''
            If the clause is not satisfied, flipping the variable will make it sat
            Not applicable here, as per lecture notes we look for no damage, hence only 
            check if the flip will not make any clause go unsat.
            
            for y in self.literals[x]:
                if self.numSatisfiedLitsPerClause[str(y)] is 0:
                    gain += 1
            '''
            '''
            If the -x is only variable the satisfies the clause, flipping it to x will 
            make the clause unsat
            '''
            for y in self.literals[x * -1]:
                if self.numSatisfiedLitsPerClause[str(y)] is 1:
                    loss +=1

            '''
            
            '''
            #candidate = gain - loss
            '''
            as per assigment brief step #2:
                "
                If at least one variable in BC has negative gain of 0 (i.e. flipping the variable does not
                make any clause that is currently satisfied go unsat), randomly select one of these
v               ariables.
                "
                adding variables with 0 negative gain to the list for random choosing
            '''
            if loss == 0:
                #index is 0 based
                position = x - 1
                gainFound = True
                ties.append(position)
            

        if gainFound:
            if len(ties) > 1:
                #break any potential tie
                index = random.randint(0, len(ties) - 1)
                self.flip(ties[index])
            else:
                #if no tie, just flip selected
                self.flip(position)
            return True
        return False


    def pickPositionAndFlip(self, C):
        
        unSatClauses = len(self.falseClauses)
        #setting the maximum to all possible clauses to be false
        maxUnsat = -unSatClauses
        ties = []
        gainFound = False
        for x in C:
            #gained sat clauses as a result of var flipping
            gain = 0
            #clauses becoming unsat as a result for the flip
            loss = 0

            #As suggested during lecture, will only look at the clauses that 
            #contain the variable

            '''
            If the clause is not satisfied, flipping the variable will make it sat
            '''
            for y in self.literals[x]:
                if self.numSatisfiedLitsPerClause[str(y)] is 0:
                    gain += 1
            '''
            If the -x is only variable the satisfies the clause, flipping it to x will 
            make the clause unsat
            '''
            for y in self.literals[x * -1]:
                if self.numSatisfiedLitsPerClause[str(y)] is 1:
                    loss +=1

            '''
            As per assignment brief (page 3): 
            "
                The best neighbour is defined as the one that has largest net gain, 
                i.e. largest value for B0 – B1 where B0 is the number of clauses that are currently unsat, 
                and B1 is the number of clauses that would be unsat after flipping the variable.
            "
            Given that the current unsat clauses is constant, I can simplyfy the above to simple
            delta calculation of net gain defined as clauses changed to true minus
            clauses changed to false, or if need to fully expand to B0 - B1, it could be expanded as follows:
                NetGain = unsatClauses - (unsatClauses - gain + loss))
            '''
            candidate = gain - loss
            #if the current is worse that the potential flip
            if maxUnsat < candidate:
                #clear any potential ties
                ties.clear()
                maxUnsat = candidate
                #index is 0 based
                position = x - 1
                gainFound = True
                ties.append(position)
            elif maxUnsat == candidate:
                ties.append(abs(x) - 1)

        if gainFound:
            if len(ties) > 1:
                #break any potential tie
                index = random.randint(0, len(ties) - 1)
                self.flip(ties[index])
            else:
                #if no tie, just flip selected
                self.flip(position)





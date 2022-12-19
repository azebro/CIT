#!/usr/bin/python

import sys
import random
import timeit
import numpy as np
import statistics
import pandas as pd




class GWSAT:
    
   
    
    '''
    Arguments:
        instance - the file to be processed
        executions - number or exections to run
        restarts - number of restarts
        iterations - number of iterations for attempted solution
        wp - the probability of random walk
    '''
    
    def __init__(self, instance, executions, restarts, iterations, wp):
        
        self.instance = instance
        self.executions = executions
        self.restarts = restarts
        self.iterations = iterations
        self.wp = wp
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
        #variables that are part of unsat clauses
        self.falseVariables = []
        


    def start(self):
        start = timeit.default_timer()
        
        runStatistics = []
        iterationsToSolve = []
        
        self.readFile(self.instance)
        self.createContainingClauses()
        #The counter of actually performend restarts
        actualRestarts = 0
        #Executions loop      
        for e in range(0, self.executions):
            #Random seed set to student id + change per execution as advised on the 
            #discussion forum
            random.seed(183247 + e * 1000)
            #Restarts loop
            for r in range(0, self.restarts):
                
                self.generateRandomTStatement()
                self.setFalseClauses()
                #Attempt to solve the SAT problem
                outcome = self.Solve(self.iterations)
                runStatistics.append(outcome)
                #If the solution is found, there is no need to keep restarting
                if outcome[0] == 1:
                    iterationsToSolve.append(r)
                    break
                actualRestarts += 1 

                        
           
        end = timeit.default_timer()
        stats = np.array(runStatistics)
       
        #print(stats)
        return stats, actualRestarts, start, end
        
        
    def Solve(self, iterations):
        solveStart = timeit.default_timer() 
        returnData = []
        isSat = 0
        flips = 0
        #Iterations (flips) loop
        for iteration in range(1, iterations  + 1):
            flips = iteration           
            r = random.random()
            '''
            Random walk with probability wp.
            if random number is less or equal wp -> random walk, otherwise GSAT
            '''
            if (r > self.wp):
                #GSAT
                self.pickPositionAndFlip()
            else:
                #Random walk - choose random variable from the ones involved in 
                #unsat clauses
                x = random.choice(self.falseVariables) - 1
                self.flip(x)
            self.setFalseClauses()
            #Check is the problem is solved, if so exit loop
            if  self.complete():
                isSat = 1
                break
                
         
        solveStop = timeit.default_timer()
        returnData = [isSat, self.truthStatement, flips, solveStop - solveStart, len(self.falseClauses)]
        return returnData
        

    '''
    Generate radom T statement, based on choice between positive and negative
    '''
    def generateRandomTStatement(self):
        self.truthStatement = []
        for x in range(1, self.variablesCount + 1):
            self.truthStatement.append(x * random.choice([-1, 1]))

    #read the instance file
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
            #list containing povitives
            self.literals[x] = []
            #list containing negatives
            self.literals[x * -1] = []
            for instance in self.clauses:
                if x in instance:
                    self.literals[x].append(instance)
                if (x * -1) in instance:
                    self.literals[x * -1].append(instance)


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

    '''
    Check is the search is complete

    '''
    def complete(self):
        if len(self.falseClauses) == 0:
            return True
        else:
            return False
    
    def flip(self,x):
        #v = []
        #Flip the variable in the statement
        self.truthStatement[x] = self.truthStatement[x] * -1
        #Process the satisfied literals to be used in calculating the
        #net gain for greedy search
        #The below can be uncommented if setFalseClauses is not used after each flip
        #The below potentially introduces the execution time reduction 
        #Baesd on the lecture comment to calculate the false clauses only once and then increment/decrement only
        '''        
        for instance in self.numSatisfiedLitsPerClause.keys():
            data = eval(instance)
            #check for the old value (not flipped)
            if str(x) in instance:
                #if more than 1 literal satisfies the clause, decrease
                if (self.numSatisfiedLitsPerClause[instance] > 0):
                    self.numSatisfiedLitsPerClause[instance] -= 1
                #if that was the only satisfying literal, add to unsat clauses after decrementing above
                if(self.numSatisfiedLitsPerClause[instance] == 0):
                    self.falseClauses.append(data)
                    v += data
            #check the flipped value
            if str(x * -1) in instance:
                #Remove the clause from the unsatisfied ones, as now there is 1 lit that satisfies it
                if(self.numSatisfiedLitsPerClause[instance] == 0):
                    self.falseClauses.remove(data)
                #Increase count of satisfied literals
                self.numSatisfiedLitsPerClause[instance] += 1
        self.falseVariables = list(set([abs(l) for l in v]))
        '''
                
      
    
    def pickPositionAndFlip(self):
        
        unSatClauses = len(self.falseClauses)
        #setting the maximum to all possible clauses to be false
        maxUnsat = -unSatClauses
        ties = []
        gainFound = False
        for x in range(1,self.variablesCount + 1):
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


   



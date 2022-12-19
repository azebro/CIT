

import numpy as np
import pandas as pd
import sys
from agents import *
import random
from mpmath import nint, sqrt

#Things
class Wolf(Thing):
    pass

class Berry(Thing):
    pass

class Mushroom(Thing):
    pass

class Grandma(Thing):
    pass


#Environment
class DarkForest(XYEnvironment):

    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        loc = copy.deepcopy(agent.location) # find out the target location
        
        #Check if agent is about to bump into a wall
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if not self.is_inbounds(loc):
            things.append(Bump())
        return things
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action == 'turnright':
            print('{} decided to {} at location: {} and is facing'.format(str(agent)[1:-1], action, agent.location, agent.direction.direction))
            agent.turn(Direction.R)
        elif action == 'turnleft':
            print('{} decided to {} at location: {} and is facing {}'.format(str(agent)[1:-1], action, agent.location, agent.direction.direction))
            agent.turn(Direction.L)
        elif action == 'moveforward':
            print('{} is thinking to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            move, why = agent.moveForward()
            if move:
                print('{} moved {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            else:
                print('{} abandoned idea to move {}wards at location: {} because {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location, why))
        elif action == "eatMushroom":
            items = self.list_things_at(agent.location, tclass=Mushroom)
            if len(items) != 0:
                if agent.eatMushroom(items[0]):
                    print('{} ate  {} at location {}'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "eatBerry":
            items = self.list_things_at(agent.location, tclass=Berry)
            if len(items) != 0:
                if agent.eatBerry(items[0]): 
                    print('{} ate  {} at location {}'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "meetWolf":
            items = self.list_things_at(agent.location, tclass=Wolf)
            if len(items) != 0:
                if agent.meetWolf(items[0]):
                    print('{} met {} at location {} and got scared, Grrrrrr'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
                else:
                    agent.stepBack()
                    print('{} met {} at location {} and stepped back'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
        elif action == "fightWolf":
            items = self.list_things_at(agent.location, tclass=Wolf)
            if len(items) != 0:
                if agent.meetWolf(items[0]):
                    print('{} met {} at location {} and killed it, Grrrrrr'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "meetGrandma":
            items = self.list_things_at(agent.location, tclass=Grandma)
            if len(items) != 0:
                if agent.meetGrandma(items[0]):
                    print('{} found a {} at location {} and is happy'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
               
                    
    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but we also send our IrishMan to sleep when the beer_threshold is crossed'''
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        complete_agents = all(agent.isComplete() for agent in self.agents)
        return dead_agents or complete_agents
#End Class


#Class Agent
class GoalWalker(Agent):
   
    

    def __init__(self, program=None):
        self.location = [0,1]
        self.direction = Direction("down")
        self.complete = False
        self.energy = 25
        self.wolves = []
        #Used if wolf is encountered and not enough energy, agent has to step back
        self.lastLocation = []
        self.lastDirection = Direction("down")
        self.visitedList = []
        self.desiredLocation = [4,4]
        self.currentDistance = self.calculateDistance(self.location[0], self.location[1], self.desiredLocation[0], self.desiredLocation[1])

        super().__init__(program)
    
    
    def moveForward(self, success=True):
        '''moveforward possible only if success (i.e. valid destination location)'''
        if not success:
            return
        self.lastLocation = copy.deepcopy(self.location)
        
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1

        if self.checkWolf():
            self.location = copy.deepcopy(self.lastLocation)
            self.turn(Direction.R)
            self.energy += 10
            return False, "Wolf was detected here before, do not step, turn right"
        
        if self.location in self.visitedList:
            self.turn(Direction.R)
            return False, "place was already visited, will trurn right"
            
        elif self.evaluateGoal():
             self.visitedList.append(copy.deepcopy(self.location))
             return True, "OK"
        else:
            self.location = copy.deepcopy(self.lastLocation)
            self.turn(Direction.R)
            return False, "Worse distance"
        
        
        self.energy -= 1
        self.check_energy()
        return True
    
    def turn(self, d):
        self.lastDirection = copy.deepcopy(self.direction)
        self.direction = self.direction + d
        

    def checkWolf(self):
        if len(self.wolves) > 0:
            if self.location in self.wolves:
                return True
        return False
    
    def stepBack(self):
        self.location = copy.deepcopy(self.lastLocation)
        self.currentDistance = self.calculateDistance(self.location[0], self.location[1], self.desiredLocation[0], self.desiredLocation[1])
        self.direction = self.lastDirection + Direction.R
        self.energy -= 1
        
    def eatBerry(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Berry):
            self.energy += 1
            return True
        return False
    
    def eatMushroom(self, thing):
        ''' returns True upon success or False otherwise'''
        if isinstance(thing, Mushroom):
            self.energy += 2
            return True
        return False
    
    def meetWolf(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Wolf) and self.energy > 15:
            self.energy -= 10
            #self.check_energy()
            self.wolves.append(copy.deepcopy(self.location))
            return True
        return False

    def meetGrandma(self, thing):
        ''' returns True upon success or False otherwise'''
        self.complete = True
        return True
    
    def isComplete(self):
        return self.complete

    def calculateDistance(self, x1, y1, x2, y2) :
        distance = nint( sqrt( (x1 - x2)**2 + (y1 - y2)**2) )
        return float(distance)
    
    def check_energy(self):
        if self.energy <= 0:
            self.happy = True
            print('Agent has run out of energy')

    def evaluateGoal(self):
        distance = self.calculateDistance(self.location[0], self.location[1], self.desiredLocation[0], self.desiredLocation[1])
        if distance <= self.currentDistance:
            self.currentDistance = distance
            return True
        return False


    
    
#End class Agent

#Program
def program(percepts):
    turn = True
    for p in percepts:
        if isinstance(p, Berry):
            return 'eatBerry'
        elif isinstance(p, Mushroom):
            return 'eatMushroom'
        elif isinstance(p, Wolf):
            return 'meetWolf'
        elif isinstance(p, Grandma):
            return 'meetGrandma'
        if isinstance(p,Bump): # then check if you are at an edge and have to turn
            turn = False
            choice = random.choice((1,2));
        elif turn:
            choice = random.choice((3,4)) # 1-right, 2-left, others-forward
        else:
             choice = random.choice((1, 2, 3, 4))
    if choice == 1:
        turn = True
        return 'turnright'
    elif choice == 2:
        turn = True
        return 'turnleft'
    else:
        turn = False
        return 'moveforward'
#end method


tp = DarkForest(5,5) # TempleBar width is set to 5, and height to 30
rrh = GoalWalker(program)

#Add volves
tp.add_thing(Wolf(), [0,2])
tp.add_thing(Wolf(), [0,4])
tp.add_thing(Wolf(), [2,1])
tp.add_thing(Wolf(), [2,3])
tp.add_thing(Wolf(), [4,2])

#Add Mushrooms
tp.add_thing(Mushroom(), [2,2])
tp.add_thing(Mushroom(), [3,4])

#Add Berries
tp.add_thing(Berry(), [0,1])
tp.add_thing(Berry(), [3,1])

#Add people
tp.add_thing(rrh, [0,0])
tp.add_thing(Grandma(), [4,4])


print("Walker starts at (1,1) facing downwards")
tp.run(50)


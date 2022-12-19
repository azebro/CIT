

import numpy as np
import pandas as pd
import sys
from agents import *
import random

#Things
class Wolf(Thing):
    pass

class Berry(Thing):
    pass

class Mushroom(Thing):
    pass

class Friend(Thing):
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
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.turn(Direction.R)
        elif action == 'turnleft':
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.turn(Direction.L)
        elif action == 'moveforward':
            print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            if agent.moveForward():
                print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            else:
                print('Move not advised')
                

        elif action == "eatMushroom":
            items = self.list_things_at(agent.location, tclass=Mushroom)
            print('Mushroom')
            if len(items) != 0:
                if agent.eatMushroom(items[0]):
                    print('{} ate {} at location {}'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
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
                    print('{} met {} at location {} and killed it, Grrrrrr'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
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
        elif action == "meetFriend":
            items = self.list_things_at(agent.location, tclass=Friend)
            if len(items) != 0:
                if agent.meetFriend(items[0]):
                    print('{} found a {} at location {} and is happy '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
                else:
                    print('{} rejects the {} at location {} and wants to have more fun '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) 
                    
    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but we also send our IrishMan to sleep when the beer_threshold is crossed'''
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        sleeping_agents = all(agent.is_sleeping() for agent in self.agents)
        return dead_agents or sleeping_agents

#End Class


#Class Agent
class ModelReflexWalker(Agent):
    location = [0,1]
    direction = Direction("down")
    sleeping = False
    energy = 25
    wolves = []
    #Used if wolf is encountered and not enough energy, agent has to step back
    lastLocation = []
    lastDirection = Direction("down")
    visitedList = []
    
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
            print("Wolf was detected here before, do not step, turn")
            return False
        if self.location in self.visitedList:
            self.turn(Direction.R)
            print("Place was already visited")
            return False
        else:
            self.visitedList.append(copy.deepcopy(self.location))

        
        
        self.energy -= 1
        self.check_energy()
        return True
    
    def turn(self, d):
        self.lastDirection = self.direction
        self.direction = self.direction + d
        self.energy -= 1
        self.check_energy()

    def checkWolf(self):
        if len(self.wolves) > 0:
            if self.location in self.wolves:
                return True
        return False
    
    def stepBack(self):
        self.location = self.lastDirection
        self.direction = self.lastDirection
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
            self.wolves.append(self.location)
            return True
        return False

    def meetFriend(self, thing):
        ''' returns True upon success or False otherwise'''
        self.sleeping = True
        return True
    
    def is_sleeping(self):
        return self.sleeping
    
    def check_energy(self):
        if self.energy <= 0:
            self.sleeping = True
            print('Agent has run out of energy')
    
    
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
        elif isinstance(p, Friend):
            return 'meetFriend'
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
man = ModelReflexWalker(program)


tp.add_thing(man, [0,1])
tp.add_thing(Berry(), [0,2])
tp.add_thing(Berry(), [0,4])
tp.add_thing(Friend(), [4,4])
tp.add_thing(Berry(), [1,1])
tp.add_thing(Mushroom(), [2,1])
tp.add_thing(Mushroom(), [3,1])
tp.add_thing(Wolf(), [4,1])
tp.add_thing(Wolf(), [4,2])
tp.add_thing(Wolf(), [4,3])
tp.add_thing(Wolf(), [3,3])
tp.add_thing(Wolf(), [0,3])

print("Walker starts at (1,1) facing downwards")
tp.run(50)


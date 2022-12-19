

import numpy as np
import pandas as pd
import sys
from agents import *

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
            move = agent.moveForward()
            if move:
                print('{} moved {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            
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
                    #self.delete_thing(items[0])
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
class ReflexWalker(Agent):
    
    
    
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
        
        super().__init__(program)
    
    
    def moveForward(self, success=True):
        '''moveforward possible only if success (i.e. valid destination location)'''
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
       
    
    def turn(self, d):
        self.direction = self.direction + d
        
        
    def eatBerry(self, thing):
        if isinstance(thing, Berry):
            return True
        return False
    
    def eatMushroom(self, thing):
        if isinstance(thing, Mushroom):
            #self.energy += 2
            return True
        return False
    
    def meetWolf(self, thing):
        if isinstance(thing, Wolf):
            return True
        return False

    def meetGrandma(self, thing):
        self.happy = True
        return True
    
    def isComplete(self):
        return self.complete
  
    
#End class Agent

#Program
def program(percepts):
    
    for p in percepts:
        if isinstance(p, Berry):
            return 'eatBerry'
        elif isinstance(p, Mushroom):
            return 'eatMushroom'
        elif isinstance(p, Wolf):
            return 'fightWolf'
        elif isinstance(p, Friend):
            return 'meetFriend'
        if isinstance(p,Bump): # then check if you are at an edge and have to turn
            turn = False
            choice = random.choice((1,2));
        else:
            choice = random.choice((1,2,3,4)) # 1-right, 2-left, others-forward
    if choice == 1:
        return 'turnright'
    elif choice == 2:
        return 'turnleft'
    else:
        return 'moveforward'
#end method



tp = DarkForest(5,5) # TempleBar width is set to 5, and height to 30
man = ReflexWalker(program)


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

print("Walker starts at (1,1) facing downwards")
tp.run(50)


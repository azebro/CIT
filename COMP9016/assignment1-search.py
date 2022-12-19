
import numpy as np
import pandas as pd
import sys
from agents import *
from search import *
import random

class RrHProblem(Problem):

    

    def __init__(self, initial, goal, obstacles, bonuses):

        self.xStart = initial[0]
        self.xEnd  = goal[1]
        self.yStart = initial[1]
        self.yEnd = goal[1]
        self.obstacles = obstacles
        self.bonuses = bonuses

        super().__init__(initial, goal)

    def isInbounds(self, location):
        """Checks to make sure that the location is inbounds (within walls if we have walls)"""
        x, y = location
        return not (x < self.xStart or x > self.xEnd or y < self.yStart or y > self.yEnd)
    def isOnObstacle(self, location):
         return location in self.obstacles
    
    def isInBonus(self, location):
        return location in self.bonuses

    def actions(self, state):
        actionList = []
        moves = [[1,1], [1,0], [0,1], [2,0], [0,2]]
        
        hasMore = True
        options = [(state[0] + 1, state[1]), (state[0], state[1] + 1), (state[0] - 1, state[1]), (state[0], state[1] - 1)]
        for option in options:
            if self.isInbounds(option): #and (not self.isOnObstacle(option)):
              actionList.append(option)
       
       
        return actionList 
    
    def result(self, state, action):
        
        return(action)

    def path_cost(self, c, state1, action, state2):
        '''
        Override to include 
            1. the bonus for mushrooms and berries
            2. penalty for wolves
        '''
        if self.isInBonus(action):
            # Zero cost, berry power
            return c
        if self.isOnObstacle(action):
            # Double cost, wolf is scary...
            return c + 2
        
        return c + 1

def report(solution):
    node, actionsExecuted = solution, []

    while node:
        actionsExecuted.append((node, node.action))
        node = node.parent

    for n in actionsExecuted[::-1]:
         print('Reached Node {} with action {}'.format(n[0], n[1]))
    print("Solution cost:   ", solution.path_cost)
    

wolves = [[0,2], [0,4], [2,1], [2,3], [4,2]]
bonuses = [[2,2], [3,4], [0,1], [3,1]]
rrh = RrHProblem((0,0), (4,4), wolves, bonuses)
#rrh.actions([1,1])
solution0 =  breadth_first_tree_search(rrh)
#solution01 = depth_first_tree_search(rrh)
solution = breadth_first_graph_search(rrh)
solution2 = depth_first_graph_search(rrh)
solution3 = depth_limited_search(rrh)

andOr = and_or_graph_search(rrh)
report(solution)
report(solution2)
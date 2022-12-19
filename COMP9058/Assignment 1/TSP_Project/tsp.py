
import numpy as np
from mpmath import nint, sqrt
import random

class NNHeuristic:
        

    def calculateDistance(self, x1, y1, x2, y2) :
        distance = nint( sqrt( (x1 - x2)**2 + (y1 - y2)**2) )
        return float(distance)

    
    def processNN(self, count, instances, data):
        """
        Implements the TSP heuristic
        Arguments:
            count {int} -- count of stops
            instances {ndarray} -- points to visit
            data {dict} -- dictionary of coordinates
        """
        id = int(random.randint(1, count))
        notVisited = instances[instances['i'] != id]['i'].tolist()
        tsp = np.array([[id, 0]])
        while notVisited:
            x = data[id][0]
            y = data[id][1]
            remaining = instances[np.isin(instances['i'], notVisited)]
            distance = np.array([remaining['i'], np.around(np.sqrt(np.power(remaining['x'] - x, 2) + np.power(remaining['y'] - y, 2)))])
            tsp = np.append(tsp, np.array([[distance[0][distance[1].argmin()], distance[1].min()]]), axis=0)
            notVisited.remove(distance[0][distance[1].argmin()])
            id = int(distance[0][distance[1].argmin()])
        #newData = {}
        ids = tsp[:, 0].astype(int)
        #for i in range(0, count):
        #    newData[i+1] = data[ids[i]][0], data[ids[i]][1] 
        return ids.tolist() # newData 



    #For testing purposes and analysis, adding as well here the proposed solution to Lab1
    """
    Author: Diarmuid Grimes, based on code of Alejandro Arbelaez
    Insertion heuristics for quickly generating (non-optimal) solution to TSP
    File contains two heuristics. 
    First heuristic inserts the closest unrouted city to the previous city 
    added to the route.
    Second heuristic inserts randomly chosen unrouted city directly after its 
    nearest city on the route
    file: lab_tsp_insertion.py
    """

    def euclideanDistane(self, cityA, cityB):
    ##Euclidean distance
    #return math.sqrt( (cityA[0]-cityB[0])**2 + (cityA[1]-cityB[1])**2 )
    ##Rounding nearest integer
        return round(sqrt( (cityA[0]-cityB[0])**2 + (cityA[1]-cityB[1])**2 ) )


    # Choose first city randomly, thereafter append nearest unrouted city to last city added to rpute
    def insertion_heuristic1(self, instance):
        cities = list(instance.keys())
        cIndex = random.randint(0, len(instance)-1)

        tCost = 0

        solution = [cities[cIndex]]
        
        del cities[cIndex]

        current_city = solution[0]
        while len(cities) > 0:
            bCity = cities[0]
            bCost = self.euclideanDistane(instance[current_city], instance[bCity])
            bIndex = 0
    #        print(bCity,bCost)
            for city_index in range(1, len(cities)):
                city = cities[city_index]
                cost = self.euclideanDistane(instance[current_city], instance[city])
    #            print(cities[city_index], "Cost: ",cost)
                if bCost > cost:
                    bCost = cost
                    bCity = city
                    bIndex = city_index
            tCost += bCost
            current_city = bCity
            solution.append(current_city)
            del cities[bIndex]
        tCost += self.euclideanDistane(instance[current_city], instance[solution[0]])
        return solution, tCost


    # Choose unrouted city randomly, insert into route after nearest routed city 
    def insertion_heuristic2(self, instance):
        cities = list(instance.keys())
        nCities=len(cities)
        cIndex = random.randint(0, len(instance)-1)

        tCost = 0

        solution = [cities[cIndex]]
        
        del cities[cIndex]

        while len(cities) > 0:
            cIndex = random.randint(0, len(cities)-1)
            nextCity = cities[cIndex]
            del cities[cIndex]
            bCost = self.euclideanDistane(instance[solution[0]], instance[nextCity])
            bIndex = 0
    #        print(nextCity,bCost)
            for city_index in range(1, len(solution)):
                city = solution[city_index]
                cost = self.euclideanDistane(instance[nextCity], instance[city])
    #            print(solution[city_index], "Cost: ",cost)
                if bCost > cost:
                    bCost = cost
                    bIndex = city_index
            solution.insert(bIndex+1, nextCity)
        for i in range(nCities):
            tCost+=self.euclideanDistane(instance[solution[i]], instance[solution[(i+1)%nCities]])
        
        return solution, tCost
        

    


        
    


        
        

        


   

    
   

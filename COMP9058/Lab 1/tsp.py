
import numpy as np
import pandas as pd
from mpmath import nint, sqrt
import random


def load_file():
    file = pd.read_csv(r"/Users/adam/CIT Repos/CIT%20MSc%20in%20AI/COMP9058/Lab1/data.tsp", header=None, skiprows=1, delimiter=r' ', names = ['i', 'x', 'y'], dtype= {"i": "int32", "x": "float", "y": "float"})
    return file

def calculate_distance(x1, y1, x2, y2) :
    distance = nint( sqrt( (x1 - x2)**2 + (y1 - y2)**2) )
    return float(distance)

def find_path(items, steps, path):
    print("Steps left: " + str(steps))
    steps = steps
    first = True
    min_distance = 0
    to_remove = 0
    node = steps.pop()
    path.append(int(node))
    c1 = items[node]
    distances = []
    remaining = steps 
    for point in remaining:
        print("Point: "+ str(point))
        if point == steps[0]: 
            print("Passing, first node")
            continue
        dis = calculate_distance(c1[0],items[point][0],c1[1],items[point][1])
        #print(dis)
        if first | (min_distance > dis):
            min_distance = dis
            first = False
            to_remove=point

    distances.append(min_distance)
    first=True 
    if len(remaining) > 1:
        print("Remaining: " +str(remaining))
        print("To move: " + str(to_remove))
        remaining.remove(to_remove)
        remaining.append(to_remove)
        distances += (find_path(items, remaining, path))
    return distances


data= load_file()

d=np.array(data)
items= {}
for index, row in data.iterrows():
    a = (row[1], row[2])
    items[row[0]]= a
    print(a)
print(items)

path = []
travel_points = list(items.keys()) 
travel_points.append(travel_points[0])
total = np.asarray(find_path(items, travel_points, path)).sum()
print("Total distance travelled: " + str(total))

output = [total] + path
np.savetxt('sol.tsp', output, fmt="%s")

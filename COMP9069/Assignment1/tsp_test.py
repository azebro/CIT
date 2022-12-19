import numpy as np


dt = np.dtype([('x', np.int16), ('y', np.int16), ('th', np.int16), ('id', np.int16)])
aa = np.array([[9 ,1, 0 , 0],[4, 5, 0, 1],[6, 10, 0, 2],[9, 10, 0, 3],[8, 9, 0, 4],[2, 7, 0, 5],[9, 3, 0, 6],[7, 9, 0, 7], [5, 6, 0, 8]] )
ss = np.array([1, 2, 3, 4]).reshape(1,4)
ff = np.concatenate((ss, aa), axis=0)
print(aa.dtype)
def calculateDistance(x1, x2, y1, y2):

    distance =  sqrt( (x1 - x2)**2 + (y1 - y2)**2) 
    return distance

def findPath(turtles):
    idx = 0
    not_visited = turtles[turtles[:,3] != idx][:,3].tolist()
    tsp_arr = np.array([[0,0]])
    while not_visited:
        currentX = turtles[turtles[:,3] == idx][0,0]
        currentY = turtles[turtles[:,3] == idx][0,1]
        remainingTurtles = turtles[np.isin(turtles[:,3], not_visited)]
        distance = np.array([
            remainingTurtles[:,3], 
            np.around(np.sqrt(np.square(remainingTurtles[:,0] - currentX) +  np.square(remainingTurtles[:,1] - currentY)))
            ])
        
        tsp_arr = np.append(tsp_arr, np.array([ [distance[0][distance[1].argmin()], distance[1].min()]]), axis=0)
        not_visited.remove(distance[0][distance[1].argmin()])
        idx = int(distance[0][distance[1].argmin()])
    return tsp_arr

bb = aa[aa[:,3] != 0]
bbb = findPath(0, aa)
print(bbb)

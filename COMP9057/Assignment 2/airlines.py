'''
Decision Analytics - Assignment 2, Task 2
Adam Zebrowski

In this task you will optimise the taxi movements for arriving and departing aircraft moving between runways and terminals. 
The input data for this task is contained in the Excel file “Assignment_DA_2_b_data.xlsx” and can be downloaded from Canvas. The file contains 3 sheets:
    - Flight schedule
    This table outlines the arrival and departure times for all flights of the day.
    - Taxi distances
    This table outlines the taxi distances between the different runways and terminals of the airport.
    - Terminal capacity
    This table shows the gate capacity of each terminal, i.e. how many planes can be present at the terminal at any given time.
The same runway cannot be occupied at the same time, neither for arrival nor for departure. 
For example, Flight B departing at 10:00 and flight L arriving at 10:00 cannot be assigned the same runway. 
Further to that, planes are occupying their allocated gate the whole timespan between arrival and departure during which the gate capacity of the terminal 
    needs to be taken into consideration when allocating terminals. 
    Planes have to taxi from the allocated arrival runway to the allocated terminal and then from the allocated terminal to the allocated departure runway. 
Arrival and departure runways can be different. The total taxi distance for each flight is the distance from the arrival runway to the allocated 
    terminal and the way back from the terminal to the departure runway.
The goal of this task is to develop and optimise an Integer Liner Programming model for allocating an arrival runway, 
    a departure runway and a terminal for each flight so that the overall taxi distance of all planes is minimised.
'''


import pandas as pd
import status as status
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

# A
'''
Load the input data from the file “Assignment_DA_2_b_data.xlsx” [1 point]. 
Make sure to use the data from the file in your code, please do not hardcode any values that can be read from the file.
'''
schedule = pd.read_excel('Assignment_DA_2_b_data.xlsx', 'Flight schedule', index_col=0)

# Extract Taxi Distace Limits
taxi = pd.read_excel('Assignment_DA_2_b_data.xlsx', 'Taxi distances', index_col=0)

# Get Terminal Capacity to fit flights
capacity = pd.read_excel('Assignment_DA_2_b_data.xlsx', 'Terminal capacity', index_col=0)

# Check the count of flights {Flight A, Flight B, Fight C.....}
all_flights = list(set(schedule.index))

# Check the count of terminals {Runway A, Runway B, Runway C.....}
all_runways = list(set(taxi.index))

# Check the count of terminals {Terminal A, Terminal B, Terminal C.....}
all_terminals = list(set(capacity.index))

print(schedule)
print('taxi',taxi)
print(capacity)
print(all_flights)
print(all_terminals)
print(all_runways)

# Create Mixed Integer Linear Program in Python using Solver
solver = pywraplp.Solver('simple mip program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# B 
'''
Identify and create the decision variables for the arrival runway allocation [1 point], 
for the departure runway allocation [1 point], 
and for the terminal allocation [1 point] using the OR Tools wrapper of the CBC_MIXED_INTEGER_PROGRAMMING solver.
'''
# (Decision Variables for Arrival Runway Allocation) ------- Eq 1
arrival_r = {}
for flight in all_flights:
    variables = {}
    for runway_a in all_runways:
        variables[runway_a] = solver.BoolVar(flight + runway_a + 'arrival')
    arrival_r[flight] = variables
#print(arrival_r)

# (Decision Variables for Departure Runway Allocation) ------- Eq 2
departure_r = {}
for flight in all_flights:
    variables = {}
    for runway_d in all_runways:
        variables[runway_d] = solver.BoolVar(flight + runway_d + 'departure')
    departure_r[flight] = variables
#print(departure_r)

# (Decision Variables for Terminal Runway Allocation) ------- Eq 3
runway_terminal = {}
for flight in all_flights:
    variables = {}
    for terminal in all_terminals:
        variables[terminal] = solver.BoolVar(flight + terminal)
    runway_terminal[flight] = variables
#print(runway_terminal)


# C 
'''
Define and create auxiliary variables for the taxi movements between runways and terminals for each flight [1 point].
'''

time_count = set()

for flight in all_flights:
    time_count.add(schedule['Arrival'][flight])
    time_count.add(schedule['Departure'][flight])

arrival_count = {}
for flight in all_flights:
    variables = {}
    for time in time_count:
        variables[str(time)] = solver.BoolVar(flight + str(time) + 'arrival')
        if schedule['Arrival'][flight] == time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    arrival_count[flight] = variables

departure_count = {}
for flight in all_flights:
    variables = {}
    for time in time_count:
        variables[str(time)] = solver.BoolVar(flight + str(time) + 'departure')
        if schedule['Departure'][flight] == time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    departure_count[flight] = variables

# D, E, F
'''
Define and implement the constraints that ensure that every flight has exactly two taxi movements [1 point].

Define and implement the constraints that ensure that the taxi movements of a flight are to and from the allocated terminal [1 point].

Define and implement the constraints that ensure that the taxi movements of a flight include the allocated arrival and departure runways [1 point].
'''
taxi_movement = {}
for flight in all_flights:
    variables = {}
    for time in time_count:
        variables[str(time)] = solver.BoolVar(flight + str(time) + 'movement')

        # E, F (constraints that ensure that the taxi movements of a flight include the allocated arrival and departure runways)
        if schedule['Departure'][flight] > time and schedule['Arrival'][flight] <= time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    taxi_movement[flight] = variables
#print(taxi_movement)


# G 
'''
Define and implement the constraints that ensure that each flight has exactly one allocated arrival runway [1 point] 
and exactly one allocated departure runway [1 point].
'''
for flight in all_flights:
    solver.Add(solver.Sum([arrival_r[flight][runway] for runway in all_runways]) == 1)
    solver.Add(solver.Sum([departure_r[flight][runway] for runway in all_runways]) == 1)

# H, J
'''
H - Define and implement the constraints the ensure that each flight is allocated to exactly one terminal [1 point].

J - Define and implement the constraints that ensure that the terminal capacities are not exceeded [1 point].
'''
for flight in all_flights:
    solver.Add(solver.Sum([runway_terminal[flight][terminal] for terminal in all_terminals]) == 1)

flight_terminal_time = {}
for terminal in all_terminals:
    for time in time_count:
        time = str(time)
        flight_terminal_time[(terminal, time)] = []
        for flight in all_flights:
            variable = solver.BoolVar(flight + terminal + 'gate' + time)
            #print( runway_terminal[flight])
            solver.Add(variable >= runway_terminal[flight][terminal] + taxi_movement[flight][str(time)] - 1)
            solver.Add(variable <= runway_terminal[flight][terminal])
            solver.Add(variable <= runway_terminal[flight][terminal])
            flight_terminal_time[(terminal, time)].append(variable)

        # J (constraints that ensure that the terminal capacities arenot exceeded)
        solver.Add(solver.Sum(flight_terminal_time[(terminal, time)]) <= int(capacity["Gates"][terminal]))

# I 
'''
Define and implement the constraints that ensure that no runway is used by more than one flight during each timeslot [1 point].
'''
for flight in all_flights:
    other = list(all_flights)
    other.remove(flight)
    for other_flight in other:
        for runway in all_runways:
            for time in time_count:
                time = str(time)
                #print(arrival_count[flight])
                solver.Add(solver.Sum([arrival_count[flight][time], arrival_count[other_flight][time],
                                       arrival_count[flight][time], arrival_count[other_flight][time]
                                       ]) <= 3)
                solver.Add(solver.Sum([arrival_count[flight][time], departure_count[other_flight][time],
                                       arrival_r[flight][runway], departure_r[other_flight][runway]
                                       ]) <= 3)
                solver.Add(solver.Sum([departure_count[flight][time], departure_count[other_flight][time],
                                       departure_r[flight][runway], departure_r[other_flight][runway]
                                       ]) <= 3)

runways_terminal_arriving = {}
for terminal in all_terminals:
    runways_terminal_arriving[terminal] = {}
    for runway in all_runways:
        runways_terminal_arriving[terminal][runway] = []
        for flight in all_flights:
            variable = solver.BoolVar("runways_terminal_arriving" + terminal + flight + runway)
            solver.Add(variable >= runway_terminal[flight][terminal] + arrival_r[flight][runway] - 1)
            solver.Add(variable <= runway_terminal[flight][terminal])
            solver.Add(variable <= arrival_r[flight][runway])
            runways_terminal_arriving[terminal][runway].append(variable)

sum_runways_terminal_arriving = {}
for terminal in all_terminals:
    sum_runways_terminal_arriving[terminal] = {}
    for runway_s in all_runways:
        sum_runways_terminal_arriving[terminal][runway_s] = solver.IntVar(0, solver.infinity(),'sum_arr' + terminal + runway_s)
        solver.Add(
            sum_runways_terminal_arriving[terminal][runway_s] == solver.Sum(runways_terminal_arriving[terminal][runway_s]))

runways_terminal_departure = {}
for terminal in all_terminals:
    runways_terminal_departure[terminal] = {}
    for runway in all_runways:
        runways_terminal_departure[terminal][runway] = []
        for flight in all_flights:
            variable = solver.BoolVar("runways_terminal_departure" + terminal + flight + runway)
            solver.Add(variable >= runway_terminal[flight][terminal] + departure_r[flight][runway] - 1)
            solver.Add(variable <= runway_terminal[flight][terminal])
            solver.Add(variable <= departure_r[flight][runway])
            runways_terminal_departure[terminal][runway].append(variable)

sum_runways_terminal_departure = {}
for terminal in all_terminals:
    sum_runways_terminal_departure[terminal] = {}
    for runway in all_runways:
        sum_runways_terminal_departure[terminal][runway] = solver.IntVar(0, solver.infinity(),
                                                                         'sum_dep' + terminal + runway)
        solver.Add(sum_runways_terminal_departure[terminal][runway] == solver.Sum(
            runways_terminal_departure[terminal][runway]))




# L
'''
Determine the arrival runway allocation [1 point], the departure runway allocation [1 point], 
and the terminal allocation [1 point] for each flight. 
Also determine the taxi distance for each flight [1 point].
'''
def assigned_runways(flights_all, runways_all, terminals_all, terminal_flight, arr, dep):
    print("F")
    for flight in sorted(flights_all):
        print("- ", flight)
        for runway in runways_all:
            if (arr[flight][runway].solution_value() == 1):
                print("    -Arival:", runway)
        for runway in runways_all:
            if (dep[flight][runway].solution_value() == 1):
                print("    -Departure:", runway)
        for terminal in terminals_all:
            if terminal_flight[flight][terminal].solution_value() == 1:
                print("    -Terminal:", terminal)


def terminal_occupied(count_time, runways_all, terminals_all, runways_terminal_arriving_sum,
                      runways_terminal_departure_sum, time_flight_terminal):
    #print("G")
    total_taxi_distance = 0
    for terminal in sorted(terminals_all):
        print("- ", terminal)
        for runway in runways_all:
            total_taxi_distance += runways_terminal_arriving_sum[terminal][runway].solution_value()
            total_taxi_distance += runways_terminal_departure_sum[terminal][runway].solution_value()

        for timed in sorted(count_time):
            total_flights_terminal = 0
            for variable in time_flight_terminal[(terminal, str(timed))]:
                total_flights_terminal += variable.solution_value()
            print("    -Time:", str(timed), "Occupied:", total_flights_terminal)
    print("- Total Taxi Distance:", total_taxi_distance)


# K
'''
Define and implement the objective function [1 point]. 
Solve the linear program and determine the optimal total taxi distances for all flights [1 point].
'''

soln = solver.Objective()

for flight in all_flights:
    for terminal in all_terminals:
        for runway in all_runways:
            soln.SetCoefficient(sum_runways_terminal_arriving[terminal][runway], float(taxi[terminal][runway]))
            soln.SetCoefficient(sum_runways_terminal_departure[terminal][runway], float(taxi[terminal][runway]))
solver.SetMinimization()
status = solver.Solve()


if status == solver.OPTIMAL:
    print("Optimal Solution")
    assigned_runways(all_flights, all_runways, all_terminals, runway_terminal, arrival_r, departure_r)
    terminal_occupied(time_count, all_runways, all_terminals, sum_runways_terminal_arriving,
                      sum_runways_terminal_departure, flight_terminal_time)
else:  # No optimal solution was found.
    if status == solver.FEASIBLE:
        print('A potentially suboptimal solution was found.')
    else:
        print('The solver could not solve the problem.')







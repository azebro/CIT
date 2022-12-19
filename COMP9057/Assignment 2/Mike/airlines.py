#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import status as status
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

# A
# Read Schedule for all flights
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

# B (Decision Variables for Arrival Runway Allocation) ------- Eq 1
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
# C (Arrival time & Departure time should be defined to identify taxi movements
# create auxiliary variables for the taxi movements between runways and terminals for each flight)
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

# D (the constraints that ensure that every flight has exactly two taxi movements)
taxi_movement = {}
for flight in all_flights:
    variables = {}
    for time in time_count:
        variables[str(time)] = solver.BoolVar(flight + str(time) + 'movement')
        # E&F (constraints that ensure that the taxi movements of a flight include the allocated arrival and departure runways)
        if schedule['Departure'][flight] > time and schedule['Arrival'][flight] <= time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    taxi_movement[flight] = variables
#print(taxi_movement)
# G (constraints that ensure that each flight has exactly one allocated arrival runway and exactly one allocated departure runway)
for flight in all_flights:
    solver.Add(solver.Sum([arrival_r[flight][runway] for runway in all_runways]) == 1)
    solver.Add(solver.Sum([departure_r[flight][runway] for runway in all_runways]) == 1)

# H (constraints that ensure that each flight is allocated to exactly one terminal)
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

# I A flight cannot share the same runway when arriving or departing with another flight
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





soln = solver.Objective()

for flight in all_flights:
    for terminal in all_terminals:
        for runway in all_runways:
            soln.SetCoefficient(sum_runways_terminal_arriving[terminal][runway], float(taxi[terminal][runway]))
            soln.SetCoefficient(sum_runways_terminal_departure[terminal][runway], float(taxi[terminal][runway]))
soln = solver.Objective()
status = solver.Solve()


# L
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


# K (Define and implement the objective function. Solve the linear program anddetermine the optimal total taxi distances for all flights)
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


# In[ ]:





# In[ ]:





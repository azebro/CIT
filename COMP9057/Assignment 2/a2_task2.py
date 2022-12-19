# Mike Leske
# R00183658

###############################################################################
#
# TASK 2 - 20 points
#
# Airport taxiway optimisation
#
# In this task you will optimise the taxi movements for arriving and departing 
# aircraft moving between runways and terminals. The input data for this task 
# is contained in the Excel file “Assignment_DA_2_b_data.xlsx” and can be 
# downloaded from Canvas. The file contains 3 sheets:
#   
#   - Flight schedule
#     This table outlines the arrival and departure times for all flights 
#     of the day.
#   - Taxi distances
#     This table outlines the taxi distances between the different runways and 
#     terminals of the airport.
#   - Terminal capacity
#     This table shows the gate capacity of each terminal, i.e. how many planes 
#     can be present at the terminal at any given time.
#
#
# The same runway cannot be occupied at the same time, neither for arrival nor 
# for departure. For example, Flight B departing at 10:00 and flight L arriving 
# at 10:00 cannot be assigned the same runway. Further to that, planes are 
# occupying their allocated gate the whole timespan between arrival and 
# departure during which the gate capacity of the terminal needs to be taken 
# into consideration when allocating terminals. Planes have to taxi from the 
# allocated arrival runway to the allocated terminal and then from the allocated 
# terminal to the allocated departure runway. Arrival and departure runways can 
# be different. The total taxi distance for each flight is the distance from the 
# arrival runway to the allocated terminal and the way back from the terminal to 
# the departure runway.
#
# The goal of this task is to develop and optimise an Integer Liner Programming 
# model for allocating an arrival runway, a departure runway and a terminal for 
# each flight so that the overall taxi distance of all planes is minimised.
#
###############################################################################

from ortools.linear_solver import pywraplp

import pandas as pd
import numpy as np
import datetime
import pprint

pp = pprint.PrettyPrinter(indent=4)

print('TASK 2')


###############################################################################
#
# TASK 2 A - 1 point
# 
#   1)  Load the input data from the file “Assignment_DA_2_b_data.xlsx” 
#       [1 point]. 
# 
#       Make sure to use the data from the file in your code, please do not 
#       hardcode any values that can be read from the file. 
#
###############################################################################

file = 'Assignment_DA_2_b_data.xlsx'

flight_schedule = pd.read_excel(file, sheet_name='Flight schedule', index_col=0).fillna(0)
taxi_distances = pd.read_excel(file, sheet_name='Taxi distances', index_col=0).fillna(0)
terminal_capacity = pd.read_excel(file, sheet_name='Terminal capacity', index_col=0).fillna(0)

flight_list     = list(flight_schedule.index)
operation_list  = list(flight_schedule.columns)
runway_list     = list(taxi_distances.index)
terminal_list   = list(terminal_capacity.index)
gates           = terminal_capacity.Gates.values


###############################################################################
#
# TASK 2 B - 3 points
# 
#   1)  Identify and create the decision variables for the arrival runway 
#       allocation [1 point], 
#   2)  for the departure runway allocation [1 point], 
#   3)  and for the terminal allocation [1 point] 
#   using the OR Tools wrapper of the CBC_MIXED_INTEGER_PROGRAMMING solver. 
#
###############################################################################

solver = pywraplp.Solver('LPWrapper', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

flights = {}
for f in flight_list:
    run_arr = [ solver.IntVar(0, 1, r.replace(' ', '') + '_' + 
                                    f.replace(' ', '') + '_arr') 
                 for r in runway_list]
    run_dep = [ solver.IntVar(0, 1, r.replace(' ', '') + '_' + 
                                    f.replace(' ', '') + '_dep') 
                 for r in runway_list]
    terminal = [ solver.IntVar(0, 1, t.replace(' ', '') + '_' + 
                                     f.replace(' ', '')) 
                 for t in terminal_list]
    time_arr = flight_schedule.loc[f, 'Arrival']
    time_dep = flight_schedule.loc[f, 'Departure']
    
    flights[f] = (run_arr, run_dep, terminal, time_arr, time_dep)


arr_slots = np.unique(flight_schedule['Arrival'].values)
dep_slots = np.unique(flight_schedule['Departure'].values)
all_slots = np.unique(np.hstack([arr_slots, dep_slots]))


def addMinutes(tm, mins):
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(minutes=mins)
    return fulldate.time()

first_slot = all_slots[0]
last_slot = addMinutes(all_slots[-1], 15)

all_slots = [ first_slot ]

while all_slots[-1] <= last_slot:
    next_slot = addMinutes(all_slots[-1], 15)
    all_slots.append(next_slot)

#pp.pprint(flights)


###############################################################################
#
# TASK 2 C - 1 point
# 
#   1)  Define and create auxiliary variables for the taxi movements between
#       runways and terminals for each flight [1 point]. 
#
###############################################################################

aux = {}
for f in flight_list:
    aux[f] = {
        'taxi_arr': {
            t: {
                r:  solver.IntVar(0, 1, 
                                  f.replace(' ', '') + '_' + 
                                  t.replace(' ', '') + '_' + 
                                  r.replace(' ', '') + '_taxi_arr')
                    for r in runway_list
            } for t in terminal_list
        },
        'taxi_dep': {
            t: {
                r:  solver.IntVar(0, 1, 
                                  f.replace(' ', '') + '_' + 
                                  t.replace(' ', '') + '_' + 
                                  r.replace(' ', '') + '_taxi_dep')
                    for r in runway_list
            } for t in terminal_list
        }
    }

#pp.pprint(aux)

###############################################################################
#
# TASK 2 D - 1 point
# 
#   1)  Define and implement the constraints that ensure that every flight has 
#       exactly two taxi movements [1 point].
#
###############################################################################

for f in flight_list:
    x = solver.Constraint(2, 2)

    for t in terminal_list:
        for r in runway_list:
            x.SetCoefficient(aux[f]['taxi_arr'][t][r], 1)
            x.SetCoefficient(aux[f]['taxi_dep'][t][r], 1)


###############################################################################
#
# TASK 2 E - 1 point
# 
#   1)  Define and implement the constraints that ensure that the taxi 
#       movements of a flight are to and from the allocated terminal [1 point].
#
###############################################################################

for f in flight_list:
    for i, t in enumerate(terminal_list):
        x_taxi_arr = solver.Constraint(0, 0)
        x_taxi_arr.SetCoefficient(flights[f][2][i], -1)

        x_taxi_dep = solver.Constraint(0, 0)
        x_taxi_dep.SetCoefficient(flights[f][2][i], -1)

        for r in runway_list: 
            x_taxi_arr.SetCoefficient(aux[f]['taxi_arr'][t][r], 1)
            x_taxi_dep.SetCoefficient(aux[f]['taxi_dep'][t][r], 1)
        

###############################################################################
#
# TASK 2 F - 1 point
# 
#   1)  Define and implement the constraints that ensure that the taxi 
#       movements of a flight include the allocated arrival and departure 
#       runways [1 point].
#
###############################################################################

for f in flight_list:

    for i, r in enumerate(runway_list):

        x_taxi_arr = solver.Constraint(0, 0)
        x_taxi_dep = solver.Constraint(0, 0)

        x_taxi_arr.SetCoefficient(flights[f][0][i], 1)
        x_taxi_dep.SetCoefficient(flights[f][1][i], 1)

        for t in terminal_list: 
            x_taxi_arr.SetCoefficient(aux[f]['taxi_arr'][t][r], -1)
            x_taxi_dep.SetCoefficient(aux[f]['taxi_dep'][t][r], -1)


###############################################################################
#
# TASK 2 G - 2 points
# 
#   1)  Define and implement the constraints that ensure that each flight has 
#       exactly one allocated arrival runway [1 point] 
#   2)  and exactly one allocated departure runway [1 point].
#
###############################################################################

for f in flight_list:
    arr_run = solver.Constraint(1, 1)
    dep_run = solver.Constraint(1, 1)
    for i, t in enumerate(runway_list):
        arr_run.SetCoefficient(flights[f][0][i], 1)
        dep_run.SetCoefficient(flights[f][1][i], 1)


###############################################################################
#
# TASK 2 H - 1 point
# 
#   1)  Define and implement the constraints the ensure that each flight is 
#       allocated to exactly one terminal [1 point].
#
###############################################################################

for f in flight_list:
    x = solver.Constraint(1, 1)
    for i, t in enumerate(terminal_list):
        x.SetCoefficient(flights[f][2][i], 1)


###############################################################################
#
# TASK 2 I - 1 point
# 
#   1)  Define and implement the constraints that ensure that no runway is used 
#       by more than one flight during each timeslot [1 point].
#
###############################################################################

for slot in all_slots:
    for i, r in enumerate(runway_list):
        x = solver.Constraint(0, 1)
        
        for f, v in flights.items():
            if v[3] == slot:
                x.SetCoefficient(v[0][i], 1)
            if v[4] == slot:
                x.SetCoefficient(v[1][i], 1)


###############################################################################
#
# TASK 2 J - 1 point
# 
#   1)  Define and implement the constraints that ensure that the terminal 
#       capacities are not exceeded [1 point].
#
###############################################################################

for slot in range(len(all_slots) - 1):
    #print(all_slots[slot] )
    for i, t in enumerate(terminal_list):
        x = solver.Constraint(0, int(gates[i]))
        
        for f, v in flights.items():
            if (v[3] <= all_slots[slot] and v[4] >= all_slots[slot+1]) or v[4] == all_slots[slot]:
                #print('  ', f, v[3], v[4])
                x.SetCoefficient(v[2][i], 1)


###############################################################################
#
# TASK 2 K - 2 points
# 
#   1)  Define and implement the objective function [1 point]. 
#   2)  Solve the linear program and determine the optimal total taxi distances 
#       for all flights [1 point].
#
###############################################################################

objective = solver.Objective()

for f in flight_list:
    for t in terminal_list:
        for r in runway_list:
            objective.SetCoefficient(aux[f]['taxi_arr'][t][r], 
                                     float(taxi_distances.loc[r, t]))
            objective.SetCoefficient(aux[f]['taxi_dep'][t][r], 
                                     float(taxi_distances.loc[r, t]))

objective.SetMinimization()
status = solver.Solve()

cost = solver.Objective().Value()

if status == solver.OPTIMAL:
    print("\nOptimal solution found")
    print("Optimal overall cost: ", cost)


###############################################################################
#
# TASK 2 L - 4 points
# 
#   1)  Determine the arrival runway allocation [1 point], 
#   2)  the departure runway allocation [1 point], 
#   3)  and the terminal allocation [1 point] for each flight. 
#   4)  Also determine the taxi distance for each flight [1 point].
#
###############################################################################

'''for f in flight_list:
    for t in terminal_list:
        for r in runway_list:
            print(aux[f]['taxi_arr'][t][r], aux[f]['taxi_arr'][t][r].solution_value())
        for r in runway_list:
            print(aux[f]['taxi_dep'][t][r], aux[f]['taxi_dep'][t][r].solution_value())
'''

print('\n####################### Task 2L ########################')
print(':Runway, Terminal allocation, taxi distance:\n')

total_taxi = 0

print('--------------------------------------------------------')
print('|   Flight   |  Arrival   |  Departure  |   Terminal   |')
print('|------------|------------|-------------|--------------|')
for f, v in flights.items():
    arr_runway = None
    dep_runway = None
    terminal   = None
    for i, r in enumerate(runway_list):
        if v[0][i].solution_value():
            arr_runway = r
    for i, r in enumerate(runway_list):
        if v[1][i].solution_value():
            dep_runway = r
    for i, t in enumerate(terminal_list):
        if v[2][i].solution_value():
            terminal = t

    total_taxi += taxi_distances.loc[arr_runway, terminal]
    total_taxi += taxi_distances.loc[dep_runway, terminal]
    #print(f, arr_runway, dep_runway, terminal)
    print('|  {}  |  {}  |  {}   |  {}  |'.format(f, arr_runway, dep_runway, terminal))
print('--------------------------------------------------------')

print('Total taxi:', total_taxi)


###############################################################################
#
# TASK 2 M - 1 point
# 
#   1)  Determine for each time of the day how many gates are occupied at each 
#       terminal [1 point].
#
###############################################################################

print('\n####################### Task 2M ########################')
print(':Gate allocation per terminal per time slot:\n')

print('---------------------------------------------------')
print('|   Slot   | Terminal A | Terminal B | Terminal C |')
print('|----------|------------|------------|------------|')
for slot in range(len(all_slots) - 1):
    used_gates = [ 0, 0, 0]

    for f, v in flights.items():
        if (v[3] <= all_slots[slot] and v[4] >= all_slots[slot+1]) or v[4] == all_slots[slot]:
            for i, t in enumerate(terminal_list):
                used_gates[i] += int(v[2][i].solution_value())
    
    #print(str(all_slots[slot]) + ' ' + str(used_gates))
    print('| {} |     {}      |     {}      |     {}      |'.format(
        all_slots[slot], used_gates[0], used_gates[1], used_gates[2]))
print('---------------------------------------------------')

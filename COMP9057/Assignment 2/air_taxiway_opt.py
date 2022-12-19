import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

#1.load data from excel file
flight_schedule=pd.read_excel('Assignment_DA_2_b_data.xlsx','Flight schedule',index_col=0)
print(flight_schedule)
taxi_distance=pd.read_excel('Assignment_DA_2_b_data.xlsx','Taxi distances',index_col=0)
print(taxi_distance)
terminal_capacity=pd.read_excel('Assignment_DA_2_b_data.xlsx','Terminal capacity',index_col=0)
print(terminal_capacity)


flights=list(set(flight_schedule.index))
terminals=list(set(terminal_capacity.index))
runways=list(set(taxi_distance.index))
times=set()

for flight in flights:
    times.add(flight_schedule['Arrival'][flight])
    times.add(flight_schedule['Departure'][flight])

print(flights)
print(terminals)
print(runways)
print(times)


solver=pywraplp.Solver('simple mip program',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

arrival_runways={}
for flight in flights:
    variables={}
    for runway in runways:
        variables[runway]=solver.BoolVar(flight+runway+'arrival')
    arrival_runways[flight]=variables

departure_runways={}
for flight in flights:
    variables={}
    for runway in runways:
        variables[runway]=solver.BoolVar(flight+runway+'departure')
    departure_runways[flight]=variables


arrival_times = {}
for flight in flights:
    variables = {}
    for time in times:
        variables[str(time)] = solver.BoolVar(flight+str(time)+'arrival')
        if flight_schedule['Arrival'][flight] == time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    arrival_times[flight] = variables

departure_times = {}
actual_departures = []
for flight in flights:
    variables={}
    for time in times:
        variables[str(time)] = solver.BoolVar(flight+str(time)+'departure')
        if flight_schedule['Departure'][flight] == time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    departure_times[flight] = variables

flight_grounded = {}
for flight in flights:
    variables = {}
    grounded_times = []
    for time in times:
        variables[str(time)] = solver.BoolVar(flight+str(time)+'grounded')
        if flight_schedule['Departure'][flight] > time and flight_schedule['Arrival'][flight] <= time:
            solver.Add(variables[str(time)] == 1)
        else:
            solver.Add(variables[str(time)] == 0)
    flight_grounded[flight] = variables


flight_terminal = {}
for flight in flights:
    variables = {}
    for terminal in terminals:
        variables[terminal] = solver.BoolVar(flight+terminal)
    flight_terminal[flight] = variables

for flight in flights:

    solver.Add(solver.Sum([arrival_runways[flight][runway] for runway in runways]) == 1)

    solver.Add(solver.Sum([departure_runways[flight][runway] for runway in runways]) == 1)

    solver.Add(solver.Sum([flight_terminal[flight][terminal] for terminal in terminals]) == 1)


flight_using_terminal_at_given_time = {}
for terminal in terminals:
    for time in times:
        time = str(time)
        flight_using_terminal_at_given_time[(terminal, time)] = []
        for flight in flights:
            variable = solver.BoolVar(flight + terminal + 'gate' + time)
            solver.Add(variable >= flight_terminal[flight][terminal]+ flight_grounded[flight][time] -1)
            solver.Add(variable <= flight_terminal[flight][terminal])
            solver.Add(variable <= flight_grounded[flight][time])
            flight_using_terminal_at_given_time[(terminal, time)].append(variable)
        solver.Add(solver.Sum(flight_using_terminal_at_given_time[(terminal, time)]) <=  int(terminal_capacity["Gates"][terminal]))


# A flight cannot share the same runway when arriving or departing with another flight
for flight in flights:
    other_flights = list(flights)
    other_flights.remove(flight)
    for other_flight in other_flights:
        for runway in runways:
            for time in times:
                time =str(time)
                solver.Add(solver.Sum([arrival_times[flight][time], arrival_times[other_flight][time],
                                       arrival_runways[flight][runway], arrival_runways[other_flight][runway]
                                       ]) <= 3)
                solver.Add(solver.Sum([arrival_times[flight][time], departure_times[other_flight][time],
                                       arrival_runways[flight][runway], departure_runways[other_flight][runway]
                                       ]) <= 3)
                solver.Add(solver.Sum([departure_times[flight][time], departure_times[other_flight][time],
                                       departure_runways[flight][runway], departure_runways[other_flight][runway]
                                       ]) <= 3)

runways_using_terminal_arriving ={}
for terminal in terminals:
    runways_using_terminal_arriving[terminal]={}
    for runway in runways:
        runways_using_terminal_arriving[terminal][runway] = []
        for flight in flights:
            variable = solver.BoolVar("runways_using_terminal_arriving"+terminal+flight+runway)
            solver.Add(variable >= flight_terminal[flight][terminal] + arrival_runways[flight][runway] - 1)
            solver.Add(variable <= flight_terminal[flight][terminal])
            solver.Add(variable <= arrival_runways[flight][runway])
            runways_using_terminal_arriving[terminal][runway].append(variable)
sum_runways_using_terminal_arriving ={}

for terminal in terminals:
    sum_runways_using_terminal_arriving[terminal]={}
    for runway in runways:
        sum_runways_using_terminal_arriving[terminal][runway] = solver.IntVar(0,solver.infinity(),'sum_arr'+terminal+runway)
        solver.Add(sum_runways_using_terminal_arriving[terminal][runway] == solver.Sum(runways_using_terminal_arriving[terminal][runway]))


runways_using_terminal_departure ={}
for terminal in terminals:
    runways_using_terminal_departure[terminal]={}
    for runway in runways:
        runways_using_terminal_departure[terminal][runway] = []
        for flight in flights:
            variable = solver.BoolVar("runways_using_terminal_departure"+terminal+flight+runway)
            solver.Add(variable >= flight_terminal[flight][terminal] + departure_runways[flight][runway] - 1)
            solver.Add(variable <= flight_terminal[flight][terminal])
            solver.Add(variable <= departure_runways[flight][runway])
            runways_using_terminal_departure[terminal][runway].append(variable)

sum_runways_using_terminal_departure ={}
for terminal in terminals:
    sum_runways_using_terminal_departure[terminal]={}
    for runway in runways:
        sum_runways_using_terminal_departure[terminal][runway] = solver.IntVar(0,solver.infinity(),'sum_dep'+terminal+runway)
        solver.Add(sum_runways_using_terminal_departure[terminal][runway] == solver.Sum(runways_using_terminal_departure[terminal][runway]))
cost = solver.Objective()

for flight in flights:
    for terminal in terminals:
        for runway in runways:
            cost.SetCoefficient(sum_runways_using_terminal_arriving[terminal][runway], float(taxi_distance[terminal][runway]))
            cost.SetCoefficient(sum_runways_using_terminal_departure[terminal][runway], float(taxi_distance[terminal][runway]))
cost = solver.Objective()
status = solver.Solve()

def F_output_runways(flights,runways,terminals,flight_terminal, arrival_runways, departure_runways):
    print("F")
    for flight in sorted(flights):
        print("- ", flight)
        for runway in runways:
            if (arrival_runways[flight][runway].solution_value() == 1):
                print("    -Arival:", runway)
        for runway in runways:
            if (departure_runways[flight][runway].solution_value() == 1):
                print("    -Departure:", runway)
        for terminal in terminals:
            if flight_terminal[flight][terminal].solution_value() == 1:
                print("    -Terminal:", terminal)

def G_terminial_occupied(times, runways,terminals,sum_runways_using_terminal_arriving,sum_runways_using_terminal_departure, flight_using_terminal_at_given_time):
    print("G")
    total_taxi_distance = 0
    for terminal in sorted(terminals):
        print("- ", terminal)
        for runway in runways:
            total_taxi_distance += sum_runways_using_terminal_arriving[terminal][runway].solution_value()
            total_taxi_distance += sum_runways_using_terminal_departure[terminal][runway].solution_value()

        for time in sorted(times):
            total_fligts_using_terminal = 0
            for variable in flight_using_terminal_at_given_time[(terminal,str(time))]:
                total_fligts_using_terminal += variable.solution_value()
            print("    -Time:",str(time),"Occupied:", total_fligts_using_terminal)
    print("- Total Taxi Distance:", total_taxi_distance)

if status == solver.OPTIMAL:
    print("Optimal Solution")
    F_output_runways(flights,runways,terminals, flight_terminal, arrival_runways,departure_runways)
    G_terminial_occupied(times, runways, terminals, sum_runways_using_terminal_arriving,
                     sum_runways_using_terminal_departure, flight_using_terminal_at_given_time)
else:  # No optimal solution was found.
    if status == solver.FEASIBLE:
        print('A potentially suboptimal solution was found.')
    else:
        print('The solver could not solve the problem.')










from ortools.sat.python import cp_model
import pandas as pd

# Reading Data from the files

'''
Load the excel file Assignment_DA_1_data.xlsx and extract all relevant information [1 point]. 
Make sure to use the data from the file in your code, please do not hardcode any values that can be read from the file.

'''

projects_data = pd.read_excel("Assignment_DA_1_data.xlsx", sheet_name="Projects", index_col=0)
projects = projects_data.index.tolist()
print(projects)
months = projects_data.columns.tolist()
print(months)
contractors_data = pd.read_excel("Assignment_DA_1_data.xlsx", sheet_name="Quotes", index_col=0)
contractors = contractors_data.index.tolist()
print(contractors)
dependencies_data = pd.read_excel("Assignment_DA_1_data.xlsx", sheet_name="Dependencies", index_col=0)
value_data = pd.read_excel("Assignment_DA_1_data.xlsx", sheet_name="Value", index_col=0)
values = value_data['Value'].tolist()
print(values)
jobs = contractors_data.columns.tolist()
print(jobs)




class solution(cp_model.CpSolverSolutionCallback):

    def __init__(self, projects_taken, contractors_project, t):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.projects_taken = projects_taken
        self.contractors_project = contractors_project
        self.profit_margin = t
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        print("Solution {}  Profit Margin: {}".format(self.__solution_count, self.Value(self.profit_margin)))

        for p in range(len(projects)):
            if self.Value(self.projects_taken[p]):
                print("{} is Taken".format(projects[p]))
                print("______________________\n")
            else:
                print("{} not taken".format(projects[p]))
                print("______________________\n")

            for m in range(len(months)):
                for c in range(len(contractors)):
                    if self.Value(self.contractors_project[(p, m, c)]):
                        print("{} , {} done by {}\n".format(months[m], projects_data.loc[projects[p]][months[m]],
                                                            contractors[c]))
        print()

    def solution_count(self):
        return self.__solution_count


def main():
    model = cp_model.CpModel()



# Creating a dictionary of contractor as key and jobs the contractor can do as values
    contractors_skill_set = {}
    for ind in contractors_data.index:
        skills = []
        for col in contractors_data.columns:
            if pd.notnull(contractors_data.loc[ind][col]):
                skills.append(col)
        contractors_skill_set[ind] = skills
    #    print(contractors_skill_set)

# Task B



# Creating decision Variables
    # (i) Identify and create the decision variables in a CP-SAT model that you need to decide what projects to take on
    projects_taken = {}
    for p in range(len(projects)):
        projects_taken[p] = model.NewBoolVar("P{}".format(projects[p].split(" ")[1]))

    # (ii) Also identify and create the decision variables you need to decide, which contractor is working on which project and when. 
    # Make sure to consider that not all contractors are qualified to work on all jobs and that projects do not run over all months
    contractor_project = {}
    for p in range(len(projects)):
        for m in range(len(months)):
            for c in range(len(contractors)):
                contractor_project[(p, m, c)] = model.NewBoolVar('p%i_m%i_c%i' % (p, m, c))
                if projects_data.loc[projects[p]][months[m]] in contractors_skill_set[contractors[c]]:
                    model.Add(contractor_project[(p, m, c)] <= 1)
                else:
                    model.Add(contractor_project[(p, m, c)] == 0)

# Task C

    # Define and implement the constraint that a contractor cannot work on two projects simultaneously

    for c in range(len(contractors)):
        for m in range(len(months)):
            model.Add(sum(contractor_project[(p, m, c)] for p in range(len(projects))) <= 1)
    #
# Task D
    #  Define and implement the constraint that if a project is accepted to be delivered then exactly one contractor per job of the project needs to work on it [4 points].
    for p in range(len(projects)):
        for m in range(len(months)):
            if pd.notnull(projects_data.loc[projects[p]][months[m]]):
                model.Add(sum(contractor_project[(p, m, c)] for c in range(len(contractors))) == 1).OnlyEnforceIf(
                    projects_taken[p])

# Task E
    # Define and implement the constraint that if a project is not taken on then no one should be contracted to work on it [4 points].
    for p in range(len(projects)):
        for m in range(len(months)):
            model.Add(sum(contractor_project[(p, m, c)] for c in range(len(contractors))) == 0).OnlyEnforceIf(
                projects_taken[p].Not())

#  Task F
    #    Define and implement the project dependency constraints [2 points].
    project_dependency = {}
    for project in projects:
        variables = {}
        for p2 in projects:
            variables[p2] = model.NewBoolVar(project + p2)
        project_dependency[project] = variables

    for i in range(len(projects)):
        for j in range(len(projects)):
            if dependencies_data[projects[i]][projects[j]] == "x":
                model.AddBoolAnd([project_dependency[projects[i]][projects[j]]])
            else:
                model.AddBoolAnd([project_dependency[projects[i]][projects[j]].Not()])

    for i in range(len(projects)):
        for j in range(len(projects)):
            model.AddBoolAnd([projects_taken[i]]).OnlyEnforceIf([project_dependency[projects[i]][projects[j]]])

# Task G
    # Define and implement the constraint that the profit margin, 
    # i.e. the difference between the value of all delivered projects and the cost of all required subcontractors, is at least €2500 [5 points].
    value_of_all_projects_delivered = []
    for i in range(len(projects)):
        value_of_all_projects_delivered.append(projects_taken[i] * values[i])
    total_value = sum(value_of_all_projects_delivered)
    cost = []
    for p in range(len(projects)):
        for c in range(len(contractors)):
            for m in range(len(months)):
                if pd.notnull(projects_data.loc[projects[p]][months[m]]):
                    job = str(projects_data.loc[projects[p]][months[m]])
                    if pd.notnull(contractors_data.loc[contractors[c]][job]) and projects_taken[p]:
                        cost.append(contractor_project[(p, m, c)] * int(contractors_data.loc[contractors[c]][job]))

    total_cost = sum(cost)
    profit_margin = total_value - total_cost
    model.Add(profit_margin >= 2500)
    t = model.NewIntVar(0, sum(values), 't')
    model.Add(t == profit_margin)

  # Task H
  # Solve the CP-SAT model and determine how many possible solutions satisfy all the constraints [1 point]. 
  # For each solution, determine what projects are taken on [1 point], 
  # which contractors work on which projects in which month [1 point], and what is the profit margin [1 point].

    solver = cp_model.CpSolver()
    solution_printer = solution(projects_taken, contractor_project, t)
    solver.SearchForAllSolutions(model, solution_printer)
    print('Number of solutions found: %i' % (solution_printer.solution_count()))





main()

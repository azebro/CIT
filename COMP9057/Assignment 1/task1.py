from ortools.sat.python import cp_model
'''
In this task you will develop a constraint satisfaction model that solves the following logical puzzle:
Carol, Elisa, Oliver and Lucas are going to university. (redundant)
One of them is going to London (1). 
Exactly one boy and one girl chose a university in a city with the same initial of their names (2). 
A boy is from Australia, the other studies History (3). 
A girl goes to Cambridge, the other studies Medicine (4). 
Oliver studies Law or is from USA; He is not from South Africa (5). 
The student from Canada is a historian or will go to Oxford (6). 
The student from South Africa is going to Edinburgh or will study Law (7). 

What is the nationality of the architecture student?
'''

# Predicates
#   nationality
#   university
#   subject
#   name
#   gender

# Attributes

names = ["Carol", "Elisa", "Lucas", "Oliver"]
nationalities = ["Australia", "South Africa", "Canada", "USA"]
universities = ["London", "Cambridge", "Oxford", "Edinburgh"]
subjects = ["Architecture", "Medicine", "History", "Law"]
genders = ["Boy", "Girl"]

# Objects
students = ["Student 1", "Student 2", "Student 3", "Student 4"]



class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, university, nationality, subject, gender, name):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.university_ = university
        self.nationality_ = nationality
        self.subject_ = subject
        self.gender_ = gender
        self.name_ = name
        self.solutions_ = 0

    def OnSolutionCallback(self):
        self.solutions_ += 1
        print(f"Solution {self.solutions_}: ")

        for student in students:
            _name = [name for name in names if self.Value(self.name_[student][name])][0]
            _university = [university for university in universities if self.Value(self.university_[student][university])][0]
            _nationality  = [nationality for nationality in nationalities if self.Value(self.nationality_[student][nationality])][0]
            _subject = [subject for subject in subjects if self.Value(self.subject_[student][subject])][0]
            print(f"| {_name}, {_nationality}, {_university}, {_subject} ")
            
        print("********************************************")


def main():
    model = cp_model.CpModel()

    student_university = {}
    student_nationalities = {}
    student_subject = {}
    student_gender = {}
    student_name = {}
    #Creating variables in a fashion of student + predicate
    for student in students:
        variables = {}
        for university in universities:
            variables[university] = model.NewBoolVar(student + "_" + university)
        student_university[student] = variables

        variables = {}
        for nationality in nationalities:
            variables[nationality] = model.NewBoolVar(student + "_" + nationality)
        student_nationalities[student] = variables

        variables = {}
        for subject in subjects:
            variables[subject] = model.NewBoolVar(student + "_" + subject)
        student_subject[student] = variables

        variables = {}
        for gender in genders:
            variables[gender] = model.NewBoolVar(student + "_" + gender)
        student_gender[student] = variables

        variables = {}
        for name in names:
            variables[name] = model.NewBoolVar(student + name)
        student_name[student] = variables

   

    # there are no 2 identical students, all have different properties
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(4):
                model.AddBoolOr([
                    student_university[students[i]][universities[k]].Not(),
                    student_university[students[j]][universities[k]].Not()])
                model.AddBoolOr([student_nationalities[students[i]][nationalities[k]].Not(),
                                 student_nationalities[students[j]][nationalities[k]].Not()])
                model.AddBoolOr([student_subject[students[i]][subjects[k]].Not(),
                                 student_subject[students[j]][subjects[k]].Not()])
                model.AddBoolOr([student_name[students[i]][names[k]].Not(),
                                 student_name[students[j]][names[k]].Not()])


    for student in students:
        #Adding constraints
        model.AddBoolOr([student_name[student]["Carol"], 
                         student_name[student]["Elisa"]]).OnlyEnforceIf(
            student_gender[student]["Girl"])

        model.AddBoolOr([student_name[student]["Oliver"], 
                         student_name[student]["Lucas"]]).OnlyEnforceIf(
            student_gender[student]["Boy"])
            
        model.AddBoolOr([student_university[student][university] for university in universities])
        model.AddBoolOr([student_nationalities[student][nationality] for nationality in nationalities])
        model.AddBoolOr([student_subject[student][subject] for subject in subjects])
        model.AddBoolOr([student_gender[student][gender] for gender in genders])
        model.AddBoolOr([student_name[student][name] for name in names])

        for i in range(4):
            for j in range(i + 1, 4):
                model.AddBoolOr([
                    student_university[student][universities[i]].Not(),
                    student_university[student][universities[j]].Not()])
                model.AddBoolOr([
                    student_nationalities[student][nationalities[i]].Not(),
                    student_nationalities[student][nationalities[j]].Not()])
                model.AddBoolOr([
                    student_subject[student][subjects[i]].Not(),
                    student_subject[student][subjects[j]].Not()])
                model.AddBoolOr([
                    student_name[student][names[i]].Not(),
                    student_name[student][names[j]].Not()])
        for i in range(2):
            for j in range(i + 1, 2):
                model.AddBoolOr([
                    student_gender[student][genders[i]].Not(),
                    student_gender[student][genders[j]].Not()])

        other_students = list(students)
        other_students.remove(student)
        # Only 1 of them is going to London: "One of them is going to London (1)."
        # university(Student a,London) => 
        #   !university(Student x,London) and !university(Student y,London) and !university(Student z,London)
        model.AddBoolAnd([student_university[other_students[0]]["London"].Not(),
                          student_university[other_students[1]]["London"].Not(),
                          student_university[other_students[2]]["London"].Not(),
                          ]).OnlyEnforceIf(student_university[student]["London"])

       
        
        for other_student in other_students:
            # "Exactly one boy and one girl chose a university in a city with the same initial of their names (2)."
            # gender(x, Boy) and gender(y,Boy) and name(x,Oliver) and university(x,Oxford) and name(y,Lucus) 
            #   => !university(y, London)
            model.AddBoolAnd([student_name[other_student]["Lucas"], 
                              student_university[other_student]["London"].Not()
                              ]).OnlyEnforceIf([student_name[student]["Oliver"],
                                                student_university[student]["Oxford"],
                                                student_gender[student]['Boy'], 
                                                student_gender[other_student]['Boy']
                                                ])
            # gender(x, Boy) and gender(y,Boy) and name(x,Lucus) and university(x,London)∧ name(y,Oliver) => !university(y, Oxford)
            model.AddBoolAnd([student_name[other_student]["Oliver"], 
                              student_university[other_student]["Oxford"].Not()
                              ]).OnlyEnforceIf([student_name[student]["Lucas"],
                                                student_university[student]["London"],
                                                student_gender[student]['Boy'], 
                                                student_gender[other_student]['Boy']
                                                ])

            # gender(x, Girl) and gender(y,Girl) and name(x,Elisa) and university(x,Edinburgh) and name(y,Carol) => !university(y, Cambridge)
            model.AddBoolAnd([student_name[other_student]["Elisa"], 
                              student_university[other_student]["Edinburgh"].Not()
                              ]).OnlyEnforceIf([student_name[student]["Carol"],
                                                student_university[student]["Cambridge"],
                                                student_gender[student]['Girl'], 
                                                student_gender[other_student]['Girl']
                                                ])
            # gender(x, Girl) and gender(y,Girl) and name(x,Carol) and university(x,Cambridge)∧ name(y,Elisa) => !university(y, Edinburgh)
            model.AddBoolAnd([student_name[other_student]["Carol"], 
                              student_university[other_student]["Cambridge"].Not()
                              ]).OnlyEnforceIf([student_name[student]["Elisa"],
                                                student_university[student]["Edinburgh"],
                                                student_gender[student]['Girl'], 
                                                student_gender[other_student]['Girl']
                                                ])

            # A boy is from Australia, the other studies History (3).
           
            # gender(x, Boy) => nationality(x, australia) or subject(y, History)
            model.AddBoolOr([student_nationalities[student]['Australia'], 
                             student_subject[student]['History']
                             ]).OnlyEnforceIf(student_gender[student]['Boy'])
            # subject(x,History)) =>  !nationality(x, australia)
            model.AddBoolAnd([student_nationalities[student]['Australia'].Not()
                              ]).OnlyEnforceIf(student_subject[student]['History'])
            # nationality(x, australia) =>  !subject(x,History)
            model.AddBoolAnd([student_subject[student]['History'].Not()
                              ]).OnlyEnforceIf(student_nationalities[student]['Australia'])

            # A girl goes to Cambridge, the other studies Medicine (4).

            # gender(x, Girl) => university(x, Cambridge) or subject(y,Medicine))
            model.AddBoolOr([student_university[student]['Cambridge'], 
                             student_subject[student]['Medicine']
                             ]).OnlyEnforceIf(student_gender[student]['Girl'])
            # subject(x,Medicine)) =>  !university(x, Cambridge)
            model.AddBoolAnd([student_university[student]['Cambridge'].Not()
                              ]).OnlyEnforceIf(student_subject[student]['Medicine'])
            # university(x, Cambridge) =>  !subject(x,Medicine)
            model.AddBoolAnd([student_subject[student]['Medicine'].Not()
                              ]).OnlyEnforceIf(student_university[student]['Cambridge'])

            # Oliver studies Law or is from USA; He is not from South Africa (5).

            # name(x, Oliver) => nationality(x, USA) or subject(x,Law))
            model.AddBoolOr([student_nationalities[student]['USA'], 
                             student_subject[student]['Law']
                             ]).OnlyEnforceIf(student_name[student]['Oliver'])
            # subject(x,Law)) =>  !nationality(x, USA)
            model.AddBoolAnd([student_nationalities[student]['USA'].Not()
                              ]).OnlyEnforceIf(student_subject[student]['Law'])
            # nationality(x, USA) =>  !subject(x,Law)
            model.AddBoolAnd([student_subject[student]['Law'].Not()
                              ]).OnlyEnforceIf(student_nationalities[student]['USA'])
            # name(x, Oliver) => !nationality(x, South Africa)
            model.AddBoolAnd([student_nationalities[student]['South Africa'].Not()
                              ]).OnlyEnforceIf(student_name[student]['Oliver'])

            # The student from Canada is a historian or will go to Oxford (6).
            
            # nationality(x, Canada) => subject(x, History) or university(x,Oxford)
            model.AddBoolOr([student_subject[student]['History'], 
                             student_university[student]["Oxford"]
                             ]).OnlyEnforceIf(student_nationalities[student]["Canada"])
            
            # nationality(x, Canada) and university(x, Oxford)  =>  !subject(x, History)
            model.AddBoolAnd([student_subject[student]['History'].Not()
                              ]).OnlyEnforceIf([student_nationalities[student]["Canada"],
                                                student_university[student]["Oxford"]
                                                   ])
            # nationality(x, Canada) and subject(x, History) =>  !university(x,Oxford)
            model.AddBoolAnd([student_university[student]['Oxford'].Not()
                              ]).OnlyEnforceIf([student_nationalities[student]["Canada"],
                                                student_subject[student]["History"]
                                                ])

            # The student from South Africa is going to Edinburgh or will study Law (7).
           
            # nationality(x, South Africa) => subject(x, Law) or university(x,Edinburgh)
            model.AddBoolOr([student_subject[student]['Law'], 
                             student_university[student]["Edinburgh"]
                             ]).OnlyEnforceIf(student_nationalities[student]["South Africa"])
            # nationality(x, South Africa) and university(x, Edinburgh)  =>  !subject(x, Law)
            model.AddBoolAnd([student_subject[student]['Law'].Not()
                              ]).OnlyEnforceIf([student_nationalities[student]["South Africa"],
                                                student_university[student]["Edinburgh"]
                                                ])
            # nationality(x, South Africa) and subject(x, Law) =>  !university(x,Edinburgh)
            model.AddBoolAnd([student_university[student]['Edinburgh'].Not()
                              ]).OnlyEnforceIf([student_nationalities[student]["South Africa"],
                                                student_subject[student]["Law"]
                                                ])

    
    model.AddBoolAnd([student_name[students[0]]["Carol"]])
    model.AddBoolAnd([student_name[students[1]]["Elisa"]])
    model.AddBoolAnd([student_name[students[2]]["Lucas"]])
    model.AddBoolAnd([student_name[students[3]]["Oliver"]])
    solver = cp_model.CpSolver()
    status = solver.SearchForAllSolutions(model, SolutionPrinter(student_university, student_nationalities, student_subject,
                                                        student_gender, student_name))
    #Only interested in opimal solution
    if status == cp_model.OPTIMAL:

        
        for student in students:
            #Get the architecture student
            if solver.Value(student_subject[student]["Architecture"]):
                for nationality in nationalities:
                    if solver.Value(student_nationalities[student][nationality]):
                        print(f"The nationality of the Architecture student is: {nationality}. Solution is: {solver.StatusName(status)}.")
                        
                    


main()
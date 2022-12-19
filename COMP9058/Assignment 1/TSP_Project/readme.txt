The invocation of the execution is as follows:

python TSP_R00183247.py instance[,instance] runs population[,population] mutationRate[,mutationRate] iterations cases

where:

1. instance[,instance] - string or list of strings
    Either a single file name with data to be processed, or coma-separated file list
    Files should be in the current folder. 
    Passing full paths will result in an error.
2. runs - integer
    Number of runs to be executed for a given configuration.
    Runs are used to calculate min/max/average fitness
3. population[,population] - integer or list of integers
    Population numbers to use for analysis. There can be a single number or multiple coma-separated.
    In case of multiples, multiple configurations will be executed
4. mutationRate[,mutationRate] - float
    Mutation rate to be used. It can be a single number (i.e.: 0.1), or a list of coma-separated
    Supplying multiple will result in mutile configurations
5. iterations - integer
    Number of iterations per run.
6. cases - list of strings
    The coma-separates list of assignment cases to be executed, i.e: 1,3,5


Program will output statistics on the screen and will generate summary files per data file in the name of: report_instance.

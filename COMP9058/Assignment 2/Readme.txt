The submission should contain the following files:

1. GWSAT.py - the implementation if GWSAT, as per question 1 of the assignment
2. WalkSAT.py - the implementation of WalkSAT with Tabu as per question 2 of the assignment
3. Zebrowski_R00183247_GWSAT.py - execution file for GWSAT, question 1
4. Zebrowski_R00183247_WalkSAT - execution file for WalkSAT, question 2
5. uf20-01.cnf - instance file
6. uf20-01.cnf - instance file
7. uf50-01.cnf - instance file
8. GWSAT_Analytics.ipynb - Jupyter notebook where I condicted additional analysis of question 1
9. WalkSAT_Analytics.ipynb - Jupyer notebook where I conducted the additional analysis of question 2
10. RTD.py -  RTD analysis as per question 3

Execution environment 
    - Python 3.7.6 64-bit 
    - Jupyter Notebook 6.0.1 (Anaconda)


Execution:

1. GWSAT:
    python Zebrowski_R00183247_GWSAT.py instance executions restarts iterations wp
    example:  python Zebrowski_R00183247_GWSAT.py uf20-01.cnf 30 10 10000 0.4 
2. WalkSAT:
    python Zebrowski_R00183247_WalkSAT.py instance executions restarts iterations wp tl
    Example: python Zebrowski_R00183247_WalkSAT.py uf20-01.cnf 30 10 1000 0.4 5

Both executions are expected to produce statistics on the screen as per screenshots in the report.
Example:

    C:\CIT MSc Repo\CIT MSc in AI\COMP9058\Assignment 2>python Zebrowski_R00183247_WalkSAT.py uf50-01.cnf 30 10 1000 0.4 5
Processing the following:
Instance: uf50-01.cnf
Executions: 30
Restarts: 10
Iterations: 1000
wp: 0.4
tl: 5

Total run time (s): 382.497
Executions: 30
Restarts executed: 300
Success rate: 0%
Run time per restart (ms):
 Max: 2404.036
 Min: 908.986
 Mean: 1273.659
 Median: 1202.438
 STDEV: 280.114
Iterations:
 Max: 0
 Min: 0
 Mean: 0.00
 Median: 0.00
 STDEV: 0.00
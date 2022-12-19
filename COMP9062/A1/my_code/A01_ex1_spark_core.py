# Databricks notebook source
# --------------------------------------------------------
#
# PYTHON PROGRAM DEFINITION
#
# The knowledge a computer has of Python can be specified in 3 levels:
# (1) Prelude knowledge --> The computer has it by default.
# (2) Borrowed knowledge --> The computer gets this knowledge from 3rd party libraries defined by others
#                            (but imported by us in this program).
# (3) Generated knowledge --> The computer gets this knowledge from the new functions defined by us in this program.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer first processes this PYTHON PROGRAM DEFINITION section of the file.
# On it, our computer enhances its Python knowledge from levels (2) and (3) with the imports and new functions
# defined in the program. However, it still does not execute anything.
#
# --------------------------------------------------------
import pyspark
import datetime
import glob
import re
from datetime import datetime

# ------------------------------------------
# FUNCTION process_line
# ------------------------------------------
def process_line(line):
    # 1. We create the output variable
    res = ()

    # 2. We get the parameter list from the line
    params_list = line.strip().split(",")

    #(00) Date => The date of the measurement. String <%Y-%m-%d %H:%M:%S> (e.g., "2013-01-01 13:00:02").
    #(01) Bus_Line => The bus line. Int (e.g., 120).
    #(02) Bus_Line_Pattern => The pattern of bus stops followed by the bus. String (e.g., "027B1001"). It can be empty (e.g., "").
    #(03) Congestion => On whether the bus is at a traffic jam (No -> 0 and Yes -> 1). Int (e.g., 0).
    #(04) Longitude => Longitude position of the bus. Float (e.g., -6.269634).
    #(05) Latitude = > Latitude position of the bus. Float (e.g., 53.360504).
    #(06) Delay => Delay of the bus in seconds (negative if ahead of schedule). Int (e.g., 90).
    #(07) Vehicle => An identifier for the bus vehicle. Int (e.g., 33304)
    #(08) Closer_Stop => An idenfifier for the closest bus stop given the current bus position. Int (e.g., 7486). It can be no bus stop, in which case it takes value -1 (e.g., -1).
    #(09) At_Stop => On whether the bus is currently at the bus stop (No -> 0 and Yes -> 1). Int (e.g., 0).

    # 3. If the list contains the right amount of parameters
    if (len(params_list) == 10):
        # 3.1. We set the right type for the parameters
        
        params_list[1] = int(params_list[1])
        params_list[3] = int(params_list[3])
        params_list[4] = float(params_list[4])
        params_list[5] = float(params_list[5])
        params_list[6] = int(params_list[6])
        params_list[7] = int(params_list[7])
        params_list[8] = int(params_list[8])
        params_list[9] = int(params_list[9])

        # 3.2. We assign res
        res = tuple(params_list)

    # 4. We return res
    return res

# ------------------------------------------
# FUNCTION is_weekday
# filtering only the weekday files
# ------------------------------------------

def is_weekday(file_date):
    """
    Takes in a date and returns the date which falls on a weekday
    """
    name = file_date[1]
    #print(name)
    date_file = re.search(r'siri.(\d{8})\d{2}.csv', name).group(1)
    _date = datetime.strptime(date_file, '%Y%m%d')
    week_no = _date.weekday()
    if week_no < 5 :
        return True
    else:
        return False
      
      
def is_week_day(date_to_process):
  d = datetime.strptime(date_to_process, "%Y-%m-%d %H:%M:%S")
  print(d)
  return d.weekday() < 5
  

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, bus_stop, bus_line, hours_list):
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    
    
    #The below code filters the dataset based on days before even loading and parsing data. 
    #This results in smaller initial dataset and thus shorter run time.
    #This assumes though that the files are names with datastamps as in the current dataset
    #inputRDD = sc.textFile(','.join([ file_name[0] for file_name in dbutils.fs.ls(my_dataset_dir) if is_weekday(file_name)]))
    
    #Alternatively load full dataset
    inputRDD = sc.textFile(my_dataset_dir) 
    

    # TO BE COMPLETED
    #If filtering the week days based on the file names, there is no need to perform that again, as the dataset contains only relevant data
    #subsetRDD = inputRDD.map(lambda row: process_line(row)).filter(lambda row: (row[1] == bus_line) & (row[8] == bus_stop) & (row[9] == 1))
    
    #Alternatively filter by all the attributes
    subsetRDD = inputRDD.map(lambda row: process_line(row)).filter(lambda row: (row[1] == bus_line) & (row[8] == bus_stop) & (row[9] == 1) & (is_week_day(row[0])))

    # key will be hour and value is a sum of the delays
    resTuple = (0,0)
    filterHourRDD = subsetRDD.map(lambda x: (x[0].split(' ', 1)[1].split(':')[0], x[6])).aggregateByKey(resTuple, lambda x,y: (x[0] + y, x[1] + 1), lambda x,y: (x[0] + y[0], x[1] + y[1]))

    # calculating the average
    solutionRDD = filterHourRDD.mapValues(lambda v: round(v[0]/v[1],2)).filter(lambda x: x[0] in hours_list).sortBy(lambda x: x[1])

    # Operation A1: 'collect' to get all results
    resVAL = solutionRDD.collect()
    
    for item in resVAL:
        print(item)

# --------------------------------------------------------
#
# PYTHON PROGRAM EXECUTION
#
# Once our computer has finished processing the PYTHON PROGRAM DEFINITION section its knowledge is set.
# Now its time to apply this knowledge.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer finally processes this PYTHON PROGRAM EXECUTION section, which:
# (i) Specifies the function F to be executed.
# (ii) Define any input parameter such this function F has to be called with.
#
# --------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    bus_stop = 279
    bus_line = 40
    hours_list = ["07", "08", "09"]

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    my_local_path = ""
    my_databricks_path = "/"

    my_dataset_dir = "FileStore/tables/my_dataset_complete"

    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir


    # 4. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, bus_stop, bus_line, hours_list)


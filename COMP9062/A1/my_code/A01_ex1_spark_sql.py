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
import pyspark.sql.functions
import datetime
import glob
import re
from datetime import datetime


# ------------------------------------------
# FUNCTION is_weekday
# filtering only the weekday files
# ------------------------------------------

def is_weekday(file_date):
    """
    Takes in a date and returns the date which falls on a weekday
    """
    date_file = re.search(r'siri.(\d{8})\d{2}.csv', file_date[1]).group(1)
    _date = datetime.strptime(date_file, '%Y%m%d')
    week_no = _date.weekday()
    if week_no < 5 :
        return True
    else:
        return False
      
def is_week_day(input_date):
  print(input_date)
  return True

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, bus_stop, bus_line, hours_list):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("date", pyspark.sql.types.TimestampType(), False),
         pyspark.sql.types.StructField("busLineID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("busLinePatternID", pyspark.sql.types.StringType(), False),
         pyspark.sql.types.StructField("congestion", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), False),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), False),
         pyspark.sql.types.StructField("delay", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("vehicleID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("closerStopID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("atStop", pyspark.sql.types.IntegerType(), False)
         ])

    # 2. Operation C2: 'read' to create the DataFrame from the dataset and the schema
    #Load the datased and filter the week days directly on the file name level
    #that way I hav initially smaller dataser that is more efficient to work with
    #of course that way I am taking dependency on the file naming convention.
    #alternatively I could load full dataset and filter dates later
    inputDF = spark.read.format("csv") \
        .option("delimiter", ",") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load([ file_name[0] for file_name in dbutils.fs.ls(my_dataset_dir) if is_weekday(file_name)])
    
    '''
    inputDF = spark.read.format("csv") \
        .option("delimiter", ",") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)
     '''

    # TO BE COMPLETED
    #I do not need to filter out the weekends, as those are handled on the file level
    subsetDF = inputDF.filter((inputDF.busLineID  == bus_line) & (inputDF.closerStopID  == bus_stop) & (inputDF.atStop  == 1))
    
    #uncomment if full datset is loaded without prefiltering files
    #subsetDF = inputDF.filter((inputDF.busLineID  == bus_line) & (inputDF.closerStopID  == bus_stop) & (is_week_day(inputDF.date)))
    hourDF = subsetDF.withColumn('hour',pyspark.sql.functions.hour(subsetDF.date))
    filterDF = hourDF.filter(hourDF.hour.isin([int(hr) for hr in hours_list])).select('hour','delay')
    solutionDF = filterDF.groupBy("hour").agg(pyspark.sql.functions.avg( "delay").alias("averageDelay")).sort("averageDelay")
    solutionDF = solutionDF.withColumn("averageDelay", pyspark.sql.functions.round(solutionDF["averageDelay"], 2))
    solutionDF = solutionDF.withColumn("hour", pyspark.sql.functions.format_string("%02d", "hour"))

    # Operation A1: 'collect' to get all results
    resVAL = solutionDF.collect()
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
    my_local_path = "../../../3_Code_Examples/L09-25_Spark_Environment/"
    my_databricks_path = "/"
    my_dataset_dir = "FileStore/tables/my_dataset_complete/"

    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, bus_stop, bus_line, hours_list)


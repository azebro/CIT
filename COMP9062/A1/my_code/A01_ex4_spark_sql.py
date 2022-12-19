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
import time
import datetime
import glob

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark, my_dataset_dir, current_time, current_stop, seconds_horizon):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), False),
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
    inputDF = spark.read.format("csv") \
        .option("delimiter", ",") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir+"*.csv")

    # TO BE COMPLETED
    current_time_object = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    seconds_horizon_object = datetime.timedelta(0, seconds_horizon)
    end_time_object = current_time_object + seconds_horizon_object

    current_time_str_object = current_time_object.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str_object = end_time_object.strftime('%Y-%m-%d %H:%M:%S')

    #filter based on start time and time horizon
    filterDF = inputDF.filter((inputDF.date > current_time_object) & (inputDF.date < end_time_object))
    filterDF.persist()

    filterhopbusDF = filterDF.filter((inputDF.closerStopID == current_stop) & (inputDF.atStop == 1)).first()
    #vehicle to hop on
    hop_bus_vehicle = filterhopbusDF[7]

    vehicleDF = filterDF.filter((filterDF.vehicleID == hop_bus_vehicle) & (filterDF.atStop == 1)).select('vehicleID', 'date', 'closerStopID')

    vehicleDF = vehicleDF.withColumn("time", vehicleDF["date"])
    vehicleDF = vehicleDF.withColumn("stop", vehicleDF["closerStopID"])
    #vehicleDF = vehicleDF.select('vehicleID', 'time', 'stop')
    vehicleDF = vehicleDF.select('vehicleID', pyspark.sql.functions.struct("time", "stop").alias("stations"))
    solutionDF = vehicleDF.groupBy('vehicleID').agg(pyspark.sql.functions.collect_list('stations').alias("stations"))


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
    current_time = "2013-01-10 08:59:59"
    current_stop = 1935
    seconds_horizon = 1800

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
    my_main(spark, my_dataset_dir, current_time, current_stop, seconds_horizon)


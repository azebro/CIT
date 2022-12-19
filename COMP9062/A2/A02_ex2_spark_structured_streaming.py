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
from pyspark.sql.functions import *
from pyspark.sql.types import *

import os
import shutil
import time


def getDay(date):
   return date[8:10]
  #return date[6:9]
udf_get_day = udf(lambda x : getDay(x))

# ------------------------------------------
# FUNCTION my_model
# ------------------------------------------
def my_model(spark, monitoring_dir, checkpoint_dir, time_step_interval, vehicle_id):
    # 1. We create the DataStreamWritter
    myDSW = None

    # 2. We set the frequency for the time steps
    my_frequency = str(time_step_interval) + " seconds"

    # 3. Operation C1: We create the DataFrame from the dataset and the schema

    # 3.1. We define the Schema of our DF.
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

    # 3.2. We use it when loading the dataset
    inputSDF = spark.readStream.format("csv") \
                               .option("delimiter", ",") \
                               .option("quote", "") \
                               .option("header", "false") \
                               .schema(my_schema) \
                               .load(monitoring_dir)

    # TO BE COMPLETED
    #Add timestamp for watermark
    inputSDF = inputSDF.withColumn("timestamp", current_timestamp())
    #Select and filter
    df = inputSDF.select("timestamp", "date", "busLineID")
    df = df.where(col("vehicleID") == vehicle_id)
    #Add extracted day
    df = df.withColumn("day",udf_get_day(col("date")))
    #Drop the full date as not needed
    df = df.drop("date")
    #Set the watermark
    df = df.withWatermark("timestamp", "0 seconds")
    #Group by day
    df = df.groupBy(window("timestamp", my_frequency, my_frequency), col("day"))
    #aggregate and sort
    df = df.agg(sort_array(collect_set("busLineID")).alias("sortedBusLineIDs"))
    #Drop window to align with desired output
    df = df.drop("window")
    
    
    #df = df.orderBy(udf(lambda x : len(x))(col("sortedBusLineIDs")))   
    solutionSDF = df.withColumn("numBusLines", size(col("sortedBusLineIDs")))
    # Operation O1: We create the DataStreamWritter, to print by console the results in complete mode
    
    # Operation O1: We create the DataStreamWritter, to print by console the results in complete mode
    myDSW = solutionSDF.writeStream\
                       .format("console") \
                       .trigger(processingTime=my_frequency) \
                       .option("checkpointLocation", checkpoint_dir) \
                       .outputMode("append")

    # We return the DataStreamWritter
    return myDSW

# ------------------------------------------
# FUNCTION get_source_dir_file_names
# ------------------------------------------
def get_source_dir_file_names(local_False_databricks_True, source_dir, verbose):
    # 1. We create the output variable
    res = []

    # 2. We get the FileInfo representation of the files of source_dir
    fileInfo_objects = []
    if local_False_databricks_True == False:
        fileInfo_objects = os.listdir(source_dir)
    else:
        fileInfo_objects = dbutils.fs.ls(source_dir)

    # 3. We traverse the fileInfo objects, to get the name of each file
    for item in fileInfo_objects:
        # 3.1. We get a string representation of the fileInfo
        file_name = str(item)

        # 3.2. If the file is processed in DBFS
        if local_False_databricks_True == True:
            # 3.2.1. We look for the pattern name= to remove all useless info from the start
            lb_index = file_name.index("name='")
            file_name = file_name[(lb_index + 6):]

            # 3.2.2. We look for the pattern ') to remove all useless info from the end
            ub_index = file_name.index("',")
            file_name = file_name[:ub_index]

        # 3.3. We append the name to the list
        res.append(file_name)
        if verbose == True:
            print(file_name)

    # 4. We sort the list in alphabetic order
    res.sort()

    # 5. We return res
    return res


# ------------------------------------------
# FUNCTION streaming_simulation
# ------------------------------------------
def streaming_simulation(local_False_databricks_True,
                         source_dir,
                         monitoring_dir,
                         time_step_interval,
                         verbose,
                         num_batches,
                         dataset_file_names
                        ):

    # 1. We check what time is it
    start = time.time()

    # 2. We set a counter in the amount of files being transferred
    count = 0

    # 3. If verbose mode, we inform of the starting time
    if (verbose == True):
        print("Start time = " + str(start))

    # 4. We transfer the files to simulate their streaming arrival.
    for file in dataset_file_names:
        # 4.1. We copy the file from source_dir to dataset_dir
        if local_False_databricks_True == False:
            shutil.copyfile(source_dir + file, monitoring_dir + file)
        else:
            dbutils.fs.cp(source_dir + file, monitoring_dir + file)

        # 4.2. If verbose mode, we inform from such transferrence and the current time.
        if (verbose == True):
            print("File " + str(count) + " transferred. Time since start = " + str(time.time() - start))

        # 4.3. We increase the counter, as we have transferred a new file
        count = count + 1

        # 4.4. We wait the desired transfer_interval until next time slot.
        time_to_wait = (start + (count * time_step_interval)) - time.time()
        if (time_to_wait > 0):
            time.sleep(time_to_wait)

    # 5. Let's try to sort out the patch for passing the last file
    if (len(dataset_file_names) > 0):
        # 5.1. We get the name again of the last file
        file = dataset_file_names[-1]

        # 5.2. We copy the file from source_dir to dataset_dir
        if local_False_databricks_True == False:
            shutil.copyfile(source_dir + file, monitoring_dir + file[:-4] + "_redundant.csv")
        else:
            dbutils.fs.cp(source_dir + file, monitoring_dir + file[:-4] + "_redundant.csv")

        # 5.3. If verbose mode, we inform from such transferrence and the current time.
        if (verbose == True):
            print("File " + str(count) + " transferred. Time since start = " + str(time.time() - start))

        # 5.4. We increase the counter, as we have transferred a new file
        count = count + 1

        # 5.5. We wait the desired transfer_interval until next time slot.
        time_to_wait = (start + (count * time_step_interval)) - time.time()
        if (time_to_wait > 0):
            time.sleep(time_to_wait)

    # 6. We wait for another time_interval
    time.sleep(time_step_interval * num_batches)

# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(spark,
            local_False_databricks_True,
            source_dir,
            monitoring_dir,
            checkpoint_dir,
            time_step_interval,
            num_batches,
            verbose,
            vehicle_id
           ):

    # 1. We get the names of the files of our dataset
    dataset_file_names = get_source_dir_file_names(local_False_databricks_True, source_dir, verbose)

    # 2. We get the DataStreamWriter object derived from the model
    dsw = my_model(spark, monitoring_dir, checkpoint_dir, time_step_interval, vehicle_id)

    # 3. We get the StreamingQuery object derived from starting the DataStreamWriter
    ssq = dsw.start()

    # 4. We simulate the streaming arrival of files (i.e., one by one) from source_dir to monitoring_dir
    streaming_simulation(local_False_databricks_True,
                         source_dir,
                         monitoring_dir,
                         time_step_interval,
                         verbose,
                         num_batches,
                         dataset_file_names
                        )

    # 5. We stop the StreamingQuery object
    try:
      ssq.stop()
    except:
      print("Thread streaming_simulation finished while Thread ssq is still computing")

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

    # 1.1 We use as many input arguments as needed
    vehicle_id = 33145

    # 1.2. We specify the time interval each of our micro-batches (files) appear for its processing.
    time_step_interval = 5

    # 1.3. We specify the num of batches
    num_batches = 4

    # 1.4. We configure verbosity during the program run
    verbose = False

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    #my_local_path = "../../../3_Code_Examples/L07-23_Spark_Environment/"
    my_databricks_path = "/"

    #source_dir = "FileStore/tables/6_Assignments/A02_ex2_micro_dataset/"
    #monitoring_dir = "FileStore/tables/6_Assignments/my_monitoring/"
    #checkpoint_dir = "FileStore/tables/6_Assignments/my_checkpoint"

    my_local_path = "../my_datasets/"
    source_dir = "A02_ex2_micro_dataset/"
    monitoring_dir = "A02_ex2_micro_dataset/my_monitoring/"
    checkpoint_dir = "A02_ex2_micro_dataset/my_checkpoint/"

    if local_False_databricks_True == False:
        source_dir = my_local_path + source_dir
        monitoring_dir = my_local_path + monitoring_dir
        checkpoint_dir = my_local_path + checkpoint_dir
    else:
        source_dir = 'FileStore/tables/a2/A02_ex2_micro_dataset/'
        monitoring_dir = "/FileStore/tables/a2/my_monitoring/"
        checkpoint_dir = "/FileStore/tables/a2/my_checkpoint/"

    # 4. We remove the directories
    if local_False_databricks_True == False:
        # 4.1. We remove the monitoring_dir
        if os.path.exists(monitoring_dir):
            shutil.rmtree(monitoring_dir)

        # 4.2. We remove the checkpoint_dir
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
    else:
        # 4.1. We remove the monitoring_dir
        dbutils.fs.rm(monitoring_dir, True)

        # 4.2. We remove the checkpoint_dir
        dbutils.fs.rm(checkpoint_dir, True)

    # 5. We re-create the directories again
    if local_False_databricks_True == False:
        # 5.1. We re-create the monitoring_dir
        os.mkdir(monitoring_dir)

        # 5.2. We re-create the checkpoint_dir
        os.mkdir(checkpoint_dir)
    else:
        # 5.1. We re-create the monitoring_dir
        dbutils.fs.mkdirs(monitoring_dir)

        # 5.2. We re-create the checkpoint_dir
        dbutils.fs.mkdirs(checkpoint_dir)

    # 6. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    spark.conf.set("spark.sql.shuffle.partitions", 4)

    # 7. We run my_main
    my_main(spark,
            local_False_databricks_True,
            source_dir,
            monitoring_dir,
            checkpoint_dir,
            time_step_interval,
            num_batches,
            verbose,
            vehicle_id
           )

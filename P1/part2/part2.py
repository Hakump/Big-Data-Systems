from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import sys

conf = SparkConf().setAppName("part2").setMaster("spark://10.10.1.1:7077") \
            .set("spark.executor.memory", "30g") \
            .set("spark.driver.memory", "30g") \
            .set("spark.executor.cores",5) \
            .set("spark.task.cpus", 1)
sc = SparkContext(conf=conf)

inputName = sys.argv[1] 
outputPath = "hdfs://10.10.1.1:9000/" + sys.argv[2]
# read the original file from hdfs
lines = sc.textFile("hdfs://10.10.1.1:9000/" + inputName)

# split each line to array and sort the file based on the combination of country code column and timestamp column       
df = lines.map(lambda line: line.split(",")).sortBy(lambda x: (x[2], x[-1]))

# save to output to local
df.coalesce(1).saveAsTextFile(outputPath)

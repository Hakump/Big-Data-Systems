from pyspark import SparkContext, SparkConf, StorageLevel

import re

# calculating the giving rank for each iteration
def averageRank(key, pair):
    targets = pair[0]  # list of targets that the source can go to
    rank = pair[1]  # 1.0 initial rank
    result = []
    # must be
    result.append((key, 0))
    targetCount = len(targets)
    averageRank = rank / targetCount  # ditribute rank evenly based on num of targets

    for target in targets:
        temp = (target, averageRank)
        result.append(temp)

    return result

def ftl(x):
    return re.search("^category:", x[0]) or not (":" in x[0]) and (re.search("^category:", x[1]) or not (":" in x[1]))

ITERATION_NUM = 3

TEST_FILE = 'test.txt'
BIG_FILE = 'hdfs://10.10.1.1:9000/enwiki-pages-articles/*'

conf = SparkConf().setAppName("task3").setMaster("spark://10.10.1.1:7077") \
            .set("spark.executor.memory", "30g") \
            .set("spark.driver.memory", "30g") \
            .set("spark.executor.cores",5) \
            .set("spark.task.cpus", 1)
sc = SparkContext(conf=conf)            

lines = sc.textFile(BIG_FILE)  
links = lines.map(lambda line: tuple(re.split("\\t", line.lower(), maxsplit=1))) \
             .filter(lambda x: ftl(x)).groupByKey()  # not (":" in x) or ("category" in x)
links.persist(StorageLevel.MEMORY_ONLY)
ranks = links.map(lambda pair: (pair[0], 1.0))  # create the ranks <key,one> RDD from the links <key, Iter> RDD

for i in range(ITERATION_NUM):
    linkRankRDD = links.join(ranks)  # joined result should be source : (targets, rank)
    # get the list of (target, rank) pair to increase the rank
    targetSum = linkRankRDD.flatMap(lambda pair: averageRank(pair[0], pair[1])) \
                           .reduceByKey(lambda x, y: x + y)  # sum up the rank for each target
    ranks = targetSum.mapValues(lambda x: 0.15 + 0.85 * x)  # assign the new rank

ranks.saveAsTextFile("hdfs://10.10.1.1:9000/task3Out")  # save to output to local


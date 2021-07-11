To run the program, run "./run.sh", 
it will run pagerank algo on the wikidataset from the root directory in hadoop with no explictly cache or partitioning, and generating output folder(task1Out) in hadoop root directory. If need to rerun the program, you need to rm the task1Out folder from hadoop by
~/hadoop-3.1.4/bin/hdfs dfs -rm -r /task1Out
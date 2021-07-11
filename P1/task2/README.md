To run the program, run "./run.sh", 
It will run pagerank algo on the wikidataset from the root directory in hadoop with customized partitioning, and generating output folder(task2Out) in hadoop root directory. If need to rerun the program, you need to rm the task2Out folder from hadoop by
"~/hadoop-3.1.4/bin/hdfs dfs -rm -r /task2Out"
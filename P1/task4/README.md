To run the program, run "./run.sh", 
It will run pagerank algo on the wikidataset from the root directory in hadoop with killed process during running, and generating output folder(task4Out) in hadoop root directory. If need to rerun the program, you need to rm the task4Out folder from hadoop by
"~/hadoop-3.1.4/bin/hdfs dfs -rm -r /task4Out"
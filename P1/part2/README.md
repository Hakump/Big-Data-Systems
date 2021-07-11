To run the python program, type the following in the shell

$HOME/spark-2.4.6-bin-hadoop2.7/bin/spark-submit part2.py [inputFileName] [outputPath]

node that both input and output path should be the absolute path in hadoop.
For example, the sample part2.sh "$HOME/spark-2.4.6-bin-hadoop2.7/bin/spark-submit part2.py export.csv output" means it reads the input from hdfs://10.10.1.1:9000/export.csv
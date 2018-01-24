#!/bin/bash

hdfs dfs -rm -r /user/$USER/gan/fl4

hadoop jar /opt/hadoop/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar \
    -input /user/$USER/gan/data4-min.txt \
    -output /user/$USER/gan/fl4 \
    -mapper mapp.py \
    -file mapp.py

#!/bin/bash
# Numbers of available CPU's
CPU_number=$(nproc --all) 
# Number of CPU's - 1 
CPU_number=$((CPU_number-1)) 
#Loop over samples
for j in {0..29..CPU_number}; do
    #Loop over dedicated CPU's
for ((i=0; i<=$CPU_number;i++));do
    #Full sub-network characterisation 
    python net_char.py -nsample $((j+i))&
    #ID of the Python process
    pids[$((i))]=$!
done
for pid in ${pids[*]}; do
    #Wait until every Python
    #process finish
    wait $pid
done
done

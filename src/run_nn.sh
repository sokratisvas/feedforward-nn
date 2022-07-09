#!/bin/bash
gcc -Wall -std=c99 ../src/linalg.c ../irisdata/iris_load.c ../src/neuralnet.c ../src/main.c -lm -o main && ./main

#!/bin/bash
gcc -Wall -std=c99 ../src/linalg.c iris_load.c -o iris_load && ./iris_load

#!/bin/bash
gcc -Wall -std=c99 ../src/linalg.c test_linalg.c -o test_linalg && ./test_linalg

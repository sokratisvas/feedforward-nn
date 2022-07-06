#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "linalg.h"

double rand_range(double lower, double upper){
    double range = upper - lower;
    return lower + (range * rand() / RAND_MAX);
}

int index_at(int row, int column, Matrix* matrix) {
    return matrix->columns * row + column;
}

void print_dims(Matrix* matrix) {
    printf("%d X %d\n", matrix->rows, matrix->columns);
}

void print_matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            printf("%f\t", matrix->data[index_at(i, j, matrix)]);
        }
        putchar('\n');
    }
}

Matrix* new_matrix(int rows, int columns) {
    Matrix* matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = malloc(rows * columns * sizeof(double));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix->data[index_at(i, j, matrix)] = 0;
        }
    }

    return matrix;
}

Matrix* new_random_matrix(int rows, int columns) {
    Matrix* matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = malloc(rows * columns * sizeof(double));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix->data[index_at(i, j, matrix)] = rand_range(0.0, 1.0);
        }
    }

    return matrix;
}

Matrix* initialise_matrix(int rows, int columns, int total, double* values) {
    assert(total == rows * columns);
    Matrix* matrix = new_matrix(rows, columns);

    for (int i = 0; i < total; i++) {
        matrix->data[i] = values[i];
    }

    return matrix;
}

void delete_matrix(Matrix* matrix) {
    free(matrix->data);
    free(matrix);
}

Matrix* add(Matrix* matrix1, Matrix* matrix2) {
    assert(matrix1->rows == matrix2->rows && "error: check row dims");
    assert(matrix1->columns == matrix2->columns && "error: check column dimension");

    Matrix* matrix = new_matrix(matrix1->rows, matrix1->columns);
    
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->data[index_at(i, j, matrix)] = matrix1->data[index_at(i, j, matrix1)] +
                                                    matrix2->data[index_at(i, j, matrix2)];
        }
    }

    return matrix;   
}

Matrix* subtract(Matrix* matrix1, Matrix* matrix2) {
    assert(matrix1->rows == matrix2->rows && "error: check row dims");
    assert(matrix1->columns == matrix2->columns && "error: check column dimension");

    Matrix* matrix = new_matrix(matrix1->rows, matrix1->columns);
    
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->data[index_at(i, j, matrix)] = matrix1->data[index_at(i, j, matrix1)] -
                                                    matrix2->data[index_at(i, j, matrix2)];
        }
    }

    return matrix;   
}

Matrix* multiply(Matrix* matrix1, Matrix* matrix2) {
    assert(matrix1->columns == matrix2->rows && "error: check row dims");

    Matrix* matrix = new_matrix(matrix1->rows, matrix2->columns);
    
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix2->columns; j++) {
            for (int k = 0; k < matrix1->columns; k++) {
                matrix->data[index_at(i, j, matrix)] += matrix1->data[index_at(i, k, matrix1)] 
                                                        * matrix2->data[index_at(k, j, matrix2)];
            }
        }
    }

    return matrix;   
}

Matrix* transpose(Matrix* matrix){
    Matrix* matrix_transpose = new_matrix(matrix->columns, matrix->rows);

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix_transpose->data[index_at(j, i, matrix_transpose)] = matrix->data[index_at(i, j, matrix)];
        }
    }

    return matrix_transpose;
}

/*
int main() {
double d4[4] = {4.7, 3.4, 3.4, 7.9};         
double d5[2] = {1.2, 1.4};          
Matrix* m1 = initialise_matrix(2, 2, 4, d4);          
Matrix* m2 = initialise_matrix(2, 1, 2, d5);          
Matrix* actual = multiply(m1, m2);
print_dims(actual);   
print_matrix(actual);

delete_matrix(m1);
delete_matrix(m2);    
delete_matrix(actual);  
    return 0;
}
*/



#ifndef LINALG_H
#define LINALG_H

typedef struct{
    double* data;
    int rows;
    int columns;
} Matrix;

double rand_range(double lower, double upper);
int index_at(int row, int column, Matrix* matrix);
void print_dims(Matrix* matrix);
void print_matrix(Matrix* matrix);
Matrix* new_matrix(int rows, int columns);
Matrix* new_random_matrix(int rows, int columns);
Matrix* initialise_matrix(int rows, int columns, int total, double* values);
void delete_matrix(Matrix* matrix);
Matrix* add(Matrix* matrix1, Matrix* matrix2);
Matrix* subtract(Matrix* matrix1, Matrix* matrix2);
Matrix* multiply(Matrix* matrix1, Matrix* matrix2);
Matrix* transpose(Matrix* matrix);

#endif

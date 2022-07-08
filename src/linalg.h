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
void copy_data(Matrix* matrix, double* values, int values_size);
Matrix* new_matrix(int rows, int columns);
Matrix* new_random_matrix(int rows, int columns);
Matrix* initialise_matrix(int rows, int columns, int total, double* values);
void delete_matrix(Matrix* matrix);
Matrix* get_column(int column, Matrix* matrix);
Matrix* add(Matrix* matrix1, Matrix* matrix2);
Matrix* subtract(Matrix* matrix1, Matrix* matrix2);
Matrix* multiply(Matrix* matrix1, Matrix* matrix2);
Matrix* transpose(Matrix* matrix);
Matrix* multiply_by_scalar(Matrix* matrix, double scalar);
int get_max_row(Matrix* matrix);
Matrix* multiply_elementwise(Matrix* matrix1, Matrix* matrix2);

#endif

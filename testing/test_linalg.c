#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "../src/linalg.h"

#define ERROR 0.001

bool is_equal(Matrix* matrix1, Matrix* matrix2) {
    assert(matrix1->rows == matrix2->rows && matrix1->columns == matrix2->columns);
    for (int i = 0; i < matrix1->columns; i++) {
        for (int j = 0; j < matrix1->columns; j++) {
            if (abs(matrix1->data[index_at(i, j, matrix1)] - matrix2->data[index_at(i, j, matrix2)]) > ERROR) {
                return false;
            }
        }
    }
    return true;
}

void test(Matrix* expected, Matrix* actual, int test_case) {
    if (is_equal(expected, actual)) {
        printf("PASSED %d\n", test_case);
    } else {
        printf("FAILED %d\n", test_case);
        printf("expected:\n");
        print_matrix(expected);
        putchar('\n');
        printf("actual:\n");
        print_matrix(actual);
        putchar('\n');
    }
}

int main() {
    double d1[4] = {1, 0, 0, 1};         
    double d2[4] = {0, 1, 1, 0};         
    Matrix* m1 = initialise_matrix(2, 2, 4, d1);          
    Matrix* m2 = initialise_matrix(2, 2, 4, d2);          
    Matrix* actual = add(m1, m2);
    double d3[4] = {1, 1, 1, 1};
    Matrix* expected = initialise_matrix(2, 2, 4, d3);
    test(expected, actual, 0);
    
    delete_matrix(m1);
    delete_matrix(m2);
    delete_matrix(actual);
    delete_matrix(expected);

    double d4[4] = {4.7, 3.4, 3.4, 7.9};         
    double d5[2] = {1.2, 1.4};         
    m1 = initialise_matrix(2, 2, 4, d4);          
    m2 = initialise_matrix(2, 1, 2, d5);          
    actual = multiply(m1, m2);
    double d6[2] = {10.4, 15.14};
    expected = initialise_matrix(2, 1, 2, d6);
    test(expected, actual, 1);
    
    delete_matrix(m1);
    delete_matrix(m2);
    delete_matrix(actual);
    delete_matrix(expected);
    
    double d7[4] = {6.6, 4.6, 3.4, 8.2};         
    double d8[4] = {1.3, 5.3, 8.7, 4};         
    m1 = initialise_matrix(1, 4, 4, d7);          
    m2 = initialise_matrix(4, 1, 4, d8);          
    actual = multiply(m1, m2);
    double d9[1] = {95.34};
    expected = initialise_matrix(1, 1, 1, d9);
    test(expected, actual, 2);
    
    delete_matrix(m1);
    delete_matrix(m2);
    delete_matrix(actual);
    delete_matrix(expected);
      
    m1 = initialise_matrix(4, 1, 4, d8);
    m2 = initialise_matrix(1, 4, 4, d7);
    actual = multiply(m1, m2);
    double d10[16] = {8.58, 5.98, 4.42, 10.66,
                     34.98, 24.38, 18.02, 43.46,
                     57.42, 40.02, 29.58, 71.34,
                     26.4, 18.4, 13.6, 32.8};
    expected = initialise_matrix(4, 4, 16, d10);
    test(expected, actual, 3);

    delete_matrix(m1);
    delete_matrix(m2);
    delete_matrix(actual);
    delete_matrix(expected);
    
    
    m1 = initialise_matrix(4, 4, 16, d10);
    actual = transpose(m1);
    double d11[16] = {8.58, 34.98, 57.42, 26.4,
               5.98, 24.38, 40.02, 18.4,
               4.42, 18.02, 29.58, 13.6,
               10.66, 43.46, 71.34, 32.8};
    expected = initialise_matrix(4, 4, 16, d11);
    test(expected, actual, 4);
    
    delete_matrix(m1);
    delete_matrix(actual);
    delete_matrix(expected);
    
    double d12[4] = {2, 5, 3.3, 8.6};
    m1 = initialise_matrix(2, 2, 4, d12);
    double d13[4] = {4, 10, 6.6, 17.2};
    expected = initialise_matrix(2, 2, 4, d13);
    actual = multiply_by_scalar(m1, 2);
    test(expected, actual, 5);

    delete_matrix(m1);
    delete_matrix(actual);
    delete_matrix(expected);
    
    return 0;
}
   

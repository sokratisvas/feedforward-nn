#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "../src/linalg.h"
#include "../irisdata/iris_load.h"

int main() {
    srand(time(NULL));

    Matrix* X_train = new_matrix(TRAIN_SIZE, 4);
    Matrix* Y_train = new_matrix(TRAIN_SIZE, 1);
    Matrix* X_test = new_matrix(TEST_SIZE, 4);
    Matrix* Y_test = new_matrix(TEST_SIZE, 1);

    load_train_test_data(X_train, Y_train, X_test, Y_test);


    return 0;
}

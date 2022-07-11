#ifndef IRIS_LOAD_H
#define IRIS_LOAD_H

#define TRAIN_SIZE 120
#define TEST_SIZE 30

enum Species {
    SETOSA, VERSICOLOR, VIRGINICA
};

void shuffle_train_data(int *data, int data_size);
void load_train_test_data(Matrix* X_train, Matrix* Y_train, Matrix* X_test, Matrix* Y_test);


#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "../src/linalg.h"
#include "../irisdata/iris_load.h"

void shuffle_train_data(int *data, int data_size) {
    for (int i = 0; i < data_size; i++) {
        int temp_idx = rand() % data_size;
        int temp = data[temp_idx];
        data[temp_idx] = data[i];
        data[i] = temp;
    }
}

void load_train_test_data(Matrix* X_train, Matrix* Y_train, Matrix* X_test, Matrix* Y_test) {
    double input_train_data[TRAIN_SIZE][4];
    double output_train_data[TRAIN_SIZE][3];   
    double input_test_data[TEST_SIZE][4];
    double output_test_data[TEST_SIZE][3];

    double sepal_length[150];
    double sepal_width[150];
    double petal_length[150];
    double petal_width[150];
    int species[150];

    FILE* file = fopen("../irisdata/Iris.csv", "r");
    assert(file != NULL && "error: can't open file");
    char c;
    char text[150][64];
    int line_number = 0;
    int line_idx = 0;
    bool found_comma = 0;

    //Skip first line
    while ((c = fgetc(file)) != '\n') {}
        
    while ((c = fgetc(file)) != EOF) {
        if (!found_comma) {
            if (c == ',') {
                found_comma = true;
            }
        } else {
            if (c == '\n') {
                text[line_number][line_idx] = '\0';
                line_number++;
                line_idx = 0;
                found_comma = false;
            } else {
                text[line_number][line_idx++] = c;
            }

        }
    }
    fclose(file);

    char sepal_len_char[4];
    char sepal_wid_char[4];
    char petal_len_char[4];
    char petal_wid_char[4];
    
    sepal_len_char[3] = '\0';
    sepal_wid_char[3] = '\0';
    petal_len_char[3] = '\0';
    petal_wid_char[3] = '\0';

    for (int i = 0; i < 150; i++) {
        for(int j = 0; j < 3; j++) {
            sepal_len_char[j] = text[i][j];
            sepal_wid_char[j] = text[i][4 + j];
            petal_len_char[j] = text[i][8 + j];
            petal_wid_char[j] = text[i][12 + j];
        }
        sepal_length[i] = (int)(sepal_len_char[0] - '0') + 0.1 * (int)(sepal_len_char[2] - '0');
        sepal_width[i] = (int)(sepal_wid_char[0] - '0') + 0.1 * (int)(sepal_wid_char[2] - '0');
        petal_length[i] = (int)(petal_len_char[0] - '0') + 0.1 * (int)(petal_len_char[2] - '0');
        petal_width[i] = (int)(petal_wid_char[0] - '0') + 0.1 * (int)(petal_wid_char[2] - '0');
        
        if (strlen(text[i]) == 27) {
            species[i] = SETOSA;
        } else if(strlen(text[i]) == 31) {
            species[i] = VERSICOLOR;
        } else {
            species[i] = VIRGINICA;
        }

        // printf("Line %d: %f, %f, %f, %f, %d\n", i + 1, sepal_length[i], sepal_width[i], 
        //                                        petal_length[i], petal_width[i], species[i]);
    }

    bool total_data[150];
    int test_data[TEST_SIZE];
    int train_data[TRAIN_SIZE];
    int train_setosa[50];
    int train_versicolor[50];
    int train_virginica[50];

    for (int i = 0; i < 50; i++) {
        train_setosa[i] = i;
        train_versicolor[i] = 50 + i;
        train_virginica[i] = 100 + i;
    }

    shuffle_train_data(train_setosa, 50);
    shuffle_train_data(train_versicolor, 50);
    shuffle_train_data(train_virginica, 50);
    
    for(int i = 0; i < TRAIN_SIZE; i++) {
        if (i <= 34) {
            train_data[i] = train_setosa[i];
        } else if (i >= 35 && i <= 69) {
            train_data[i] = train_versicolor[i - 35];
        } else{
            train_data[i] = train_virginica[i - 70];
        }
    }
    
    shuffle_train_data(train_data, TRAIN_SIZE);
    
    for (int i = 0; i < 150; i++) {
        total_data[i] = 0;
    }

    for (int i = 0; i < 150; i++) {
        total_data[train_data[i]] = 1;
    }
    
    int test_idx = 0;
    for (int i = 0; i < 150; i++) {
        if (!total_data[i]) {
            test_data[test_idx] = i;
            test_idx++;
        }
    }

    shuffle_train_data(test_data, TEST_SIZE);

    for (int i = 0; i < TRAIN_SIZE; i++) {
        input_train_data[i][0] = sepal_length[train_data[i]];
        input_train_data[i][1] = sepal_width[train_data[i]]; 
        input_train_data[i][2] = petal_length[train_data[i]];
        input_train_data[i][3] = petal_width[train_data[i]];
        
        for (int j = 0; j < 3; j++) {
            output_train_data[i][j] = (j == species[train_data[i]]) ? 1:0;
        }
    }

    for (int i = 0; i < TEST_SIZE; i++) {
        input_test_data[i][0] = sepal_length[test_data[i]];
        input_test_data[i][1] = sepal_width[test_data[i]]; 
        input_test_data[i][2] = petal_length[test_data[i]];
        input_test_data[i][3] = petal_width[test_data[i]];

        for (int j = 0; j < 3; j++) {
            output_test_data[i][j] = (j == species[test_data[i]]) ? 1:0;
        }
    }

    double input_train_vectorized[4 * TRAIN_SIZE];
    double input_test_vectorized[4 * TEST_SIZE];
    double output_train_vectorized[3 * TRAIN_SIZE];
    double output_test_vectorized[3 * TEST_SIZE];
    
    int len = 0;
    for (int i = 0; i < TRAIN_SIZE; i++) {
        for (int j = 0; j < 4; j++) {
            input_train_vectorized[len++] = input_train_data[i][j];
        }
    }

    len = 0;
    for (int i = 0; i < TRAIN_SIZE; i++) {
        for (int j = 0; j < 3; j++) {
            output_train_vectorized[len++] = output_train_data[i][j];
        }
    }

    len = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < 4; j++) {
            input_test_vectorized[len++] = input_test_data[i][j];
        }
    }

    len = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < 3; j++) {
            output_test_vectorized[len++] = output_test_data[i][j];
        }
    }

    copy_data(X_train, input_train_vectorized, 4 * TRAIN_SIZE);
    copy_data(Y_train, output_train_vectorized, 3 * TRAIN_SIZE);
    copy_data(X_test, input_test_vectorized, 4 * TEST_SIZE);
    copy_data(Y_test, output_test_vectorized, 3 * TEST_SIZE);
}

/*
int main() {
    double input_train_data[TRAIN_SIZE][4];
    int output_train_data[TRAIN_SIZE];
    double input_test_data[TEST_SIZE][4];
    int output_test_data[TEST_SIZE];
    srand(time(NULL));
    load_train_test_data(input_train_data, output_train_data, input_test_data, output_test_data);
    return 0;
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include "../src/linalg.h"

#define TRAIN_DATA 105
#define TEST_DATA 45

enum Species {
    SETOSA, VERSICOLOR, VIRGINICA
};

int main() {
    
    //Matrix* train_input = new_matrix(150, 4);
    //Matrix* train_output = new_matrix(150, 1);

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

        printf("Line %d: %f, %f, %f, %f, %d\n", i + 1, sepal_length[i], sepal_width[i], 
                                                petal_length[i], petal_width[i], species[i]);
    }

    return 0;
}

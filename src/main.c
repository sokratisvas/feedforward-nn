#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include "../src/linalg.h"
#include "../irisdata/iris_load.h"
#include "../src/neuralnet.h"

int main() {
    srand(time(NULL));

    int neurons_per_layer[3] = {INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE};
    NeuralNet* neuralnet = new_neuralnet(neurons_per_layer, LEARNING_RATE);
    int correct_predictions = 0;

    Matrix* X_train = new_matrix(TRAIN_SIZE, 4);
    Matrix* Y_train = new_matrix(TRAIN_SIZE, 3);
    Matrix* X_test = new_matrix(TEST_SIZE, 4);
    Matrix* Y_test = new_matrix(TEST_SIZE, 3);
    
    load_train_test_data(X_train, Y_train, X_test, Y_test);
    X_train = transpose(X_train);
    Y_train = transpose(Y_train);
    X_test = transpose(X_test);
    Y_test = transpose(Y_test);

    X_train = normalize(X_train);
    X_test = normalize(X_test);

    for (int j = 0; j < EPOCHS; j++) {
        for (int i = 0; i < TRAIN_SIZE; i++) {
        // Feedforward
        Matrix* input = get_column(i, X_train);
        Matrix* output = get_column(i, Y_train);
        
        Matrix* prefirst_layer_output = get_prefirst_layer_output(neuralnet, input);
        Matrix* first_layer_output = get_first_layer_output(neuralnet, input);
        Matrix* calculated_output = get_second_layer_output(neuralnet, first_layer_output); 
       
        /*
        if(i == 0 && j == 0) {
            print_matrix(prefirst_layer_output);
            printf("------------------------\n");
            print_matrix(first_layer_output);
        }
        */

        // Cost
        double cost = means_squared_method(calculated_output, output);
        int species_prediction = get_max_row(calculated_output);
        int actual_species = get_max_row(output);
        correct_predictions += (species_prediction == actual_species);

        if (i == 0 && j % 10 == 0) {
            printf("Epoch: %d, cost = %lf\n", j, cost);
        }

        // Backprop
        Matrix* delta_second_layer = get_delta_second_layer(calculated_output, output);
        neuralnet->weights[1] = add(neuralnet->weights[1], get_delta_second_weights(neuralnet, delta_second_layer, first_layer_output));
        neuralnet->biases[1] = add(neuralnet->biases[1], get_delta_second_biases(neuralnet, delta_second_layer));

        Matrix* delta_first_layer = get_delta_first_layer(neuralnet, prefirst_layer_output, delta_second_layer);
        neuralnet->weights[0] = add(neuralnet->weights[0], get_delta_first_weights(neuralnet, delta_first_layer, input));
        neuralnet->biases[0] = add(neuralnet->biases[0], get_delta_first_biases(neuralnet, delta_first_layer));

        //printf("Correct Predictions = %d\n", correct_predictions);
        }
        
        if(j == EPOCHS - 1) {
            printf("accuracy = %lf\n", 1.0 * correct_predictions / (EPOCHS * TRAIN_SIZE));
        }
    }

    correct_predictions = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        Matrix* input = get_column(i, X_test);
        Matrix* output = get_column(i, Y_test);
        
        Matrix* first_layer_output = get_first_layer_output(neuralnet, input);
        Matrix* calculated_output = get_second_layer_output(neuralnet, first_layer_output); 
        
        double cost = means_squared_method(calculated_output, output);
        // printf("Rep %d, Cost %lf\n", i, cost);
        int species_prediction = get_max_row(calculated_output);
        int actual_species = get_max_row(output);
        correct_predictions += (species_prediction == actual_species);

    }

    printf("Correct Predictions = %d\n", correct_predictions);
    printf("Accuracy = %lf\n", 1.0 * correct_predictions / TEST_SIZE);

    return 0;
}

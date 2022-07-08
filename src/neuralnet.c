#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include "../src/linalg.h"
#include "../irisdata/iris_load.h"
#include "../src/neuralnet.h"

NeuralNet* new_neuralnet(int* neurons_per_layer, double learning_rate) {          
    NeuralNet* neuralnet = malloc(sizeof(NeuralNet));
    neuralnet->neurons_per_layer = neurons_per_layer;
    neuralnet->weights = malloc(sizeof(Matrix) * 2);
    neuralnet->biases = malloc(sizeof(Matrix) * 2);
 
    for (int i = 0; i < 2; i++) {
        neuralnet->weights[i] = new_random_matrix(neurons_per_layer[i + 1], neurons_per_layer[i]);
        neuralnet->biases[i] = new_random_matrix(neurons_per_layer[i + 1], 1);
    }
 
    neuralnet->learning_rate = learning_rate;
    return neuralnet;
}

void delete_neuralnet(NeuralNet* neuralnet) {
    for (int i = 0; i < 2; i++) {
        free(neuralnet->weights[i]);
        free(neuralnet->biases[i]);
    }
    free(neuralnet->weights);
    free(neuralnet->biases);
    free(neuralnet);
}

Matrix* activate_matrix(Matrix* matrix) {
    Matrix* res = new_matrix(matrix->rows, matrix->columns);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            res->data[index_at(i, j, res)] = tanh(matrix->data[index_at(i, j, matrix)]);
        }
    }
    return res;
}

Matrix* der_activate_matrix(Matrix* matrix) {
    Matrix* res = new_matrix(matrix->rows, matrix->columns);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            res->data[index_at(i, j, res)] = 1 - tanh(matrix->data[index_at(i, j, matrix)]) * tanh(matrix->data[index_at(i, j, matrix)]);
        }
    }
    return res;
}

Matrix* softmax(Matrix* matrix) {
    assert(matrix->columns == 1);
    Matrix* res = new_matrix(matrix->rows, 1);    
    double exp_sum = 0;
    for (int i = 0; i < matrix->rows; i++) {
        exp_sum += exp(matrix->data[i]);
    }

    for (int i = 0; i < matrix->rows; i++) {
        res->data[i] = exp(matrix->data[i]) / exp_sum;
    }
    return res;
}

Matrix* get_prefirst_layer_output(NeuralNet* neuralnet, Matrix* input) {
    assert(input->rows == neuralnet->neurons_per_layer[0] && input->columns == 1);
    Matrix* first_layer_output = add(multiply(neuralnet->weights[0], input), neuralnet->biases[0]);
    assert(first_layer_output->rows == neuralnet->neurons_per_layer[1] && first_layer_output->columns == 1);
    return first_layer_output;
}

Matrix* get_first_layer_output(NeuralNet* neuralnet, Matrix* input) {
    return activate_matrix(get_prefirst_layer_output(neuralnet, input));
}

Matrix* get_second_layer_output(NeuralNet* neuralnet, Matrix* act_first_layer_output) {
    Matrix* second_layer_output = add(multiply(neuralnet->weights[1], act_first_layer_output), neuralnet->biases[1]);
    return activate_matrix(second_layer_output);
}

/*
void forward(NeuralNet* neuralnet, Matrix* input, Matrix* calculated_output) {
    assert(input->rows == neuralnet->neurons_per_layer[0] && input->columns == 1);
    Matrix* first_layer_output = add(multiply(neuralnet->weights[0], input), neuralnet->biases[0]);
    assert(first_layer_output->rows == neuralnet->neurons_per_layer[1] && first_layer_output->columns == 1);
    Matrix* act_first_layer_output = activate_matrix(first_layer_output);
    Matrix* second_layer_output = add(multiply(neuralnet->weights[1], act_first_layer_output), neuralnet->biases[1]);
    softmax(second_layer_output, calculated_output);
}
*/

double means_squared_method(Matrix* calculated_output, Matrix* output) {
    assert(output->columns == 1 && calculated_output->columns == 1);
    assert(calculated_output->rows == output->rows);
    double error = 0;
    for (int i = 0; i < calculated_output->rows; i++) {
          error += (calculated_output->data[i] - output->data[i]) * (calculated_output->data[i] - output->data[i]);
    }
    return error / calculated_output->rows;
}

Matrix* get_delta_second_layer(Matrix* calculated_output, Matrix* output) {
   return subtract(calculated_output, output); 
}

Matrix* get_delta_second_weights(NeuralNet* neuralnet, Matrix* delta_second_layer, Matrix* first_layer_output) {
    return multiply_by_scalar(multiply(delta_second_layer, transpose(first_layer_output)), -neuralnet->learning_rate);
}

Matrix* get_delta_second_biases(NeuralNet* neuralnet, Matrix* delta_second_layer) {
    return multiply_by_scalar(delta_second_layer, -neuralnet->learning_rate);
}

Matrix* get_delta_first_layer(NeuralNet* neuralnet, Matrix* first_layer_output, Matrix* delta_second_layer) {
    return multiply_elementwise(multiply(transpose(neuralnet->weights[1]), delta_second_layer), der_activate_matrix(first_layer_output));
}

Matrix* get_delta_first_weights(NeuralNet* neuralnet, Matrix* delta_first_layer, Matrix* input) {
    return multiply_by_scalar(multiply(delta_first_layer, transpose(input)), -neuralnet->learning_rate);
}

Matrix* get_delta_first_biases(NeuralNet* neuralnet, Matrix* delta_first_layer) {
    return multiply_by_scalar(delta_first_layer, -neuralnet->learning_rate);
}

// void backward() {}

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
    
    int iter = 0;
    for (int j = 0; j < 100; j++) {
        for (int i = 0; i < TRAIN_SIZE; i++) {
        iter++;
        // Feedforward
        Matrix* input = get_column(i, X_train);
        Matrix* output = get_column(i, Y_train);
        
        Matrix* prefirst_layer_output = get_prefirst_layer_output(neuralnet, input);
        Matrix* first_layer_output = get_first_layer_output(neuralnet, input);
        Matrix* calculated_output = get_second_layer_output(neuralnet, first_layer_output); 
       
        // Cost
        double cost = means_squared_method(calculated_output, output);
        int species_prediction = get_max_row(calculated_output);
        int actual_species = get_max_row(output);
        correct_predictions += (species_prediction == actual_species);
        
        if (i == 0 && j % 10 == 0) {
            printf("Epoch: %d, cost = %lf, accuracy = %lf\n", j, cost, 1.0 * correct_predictions / iter);
        }

        // Backprop
        Matrix* delta_second_layer = get_delta_second_layer(calculated_output, output);
        neuralnet->weights[1] = add(neuralnet->weights[1], get_delta_second_weights(neuralnet, delta_second_layer, prefirst_layer_output));
        neuralnet->biases[1] = add(neuralnet->biases[1], get_delta_second_biases(neuralnet, delta_second_layer));

        Matrix* delta_first_layer = get_delta_first_layer(neuralnet, prefirst_layer_output, delta_second_layer);
        neuralnet->weights[0] = add(neuralnet->weights[0], get_delta_first_weights(neuralnet, delta_first_layer, input));
        neuralnet->biases[0] = add(neuralnet->biases[0], get_delta_first_biases(neuralnet, delta_first_layer));
    
        //printf("Correct Predictions = %d\n", correct_predictions);
    }

    
    }
    
    //printf("Correct Predictions = %d\n", correct_predictions);
    //printf("Accuracy = %lf\n", 1.0 * correct_predictions / TRAIN_SIZE);
    

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
    
    /*
    // Feedforward
    Matrix* input = get_column(0, X_train);
    Matrix* output = get_column(0, Y_train);
    
    Matrix* first_layer_output = get_first_layer_output(neuralnet, input);
    Matrix* calculated_output = get_second_layer_output(neuralnet, first_layer_output); 
    
    // Cost
    double cost = means_squared_method(calculated_output, output);
    int species_prediction = get_max_row(calculated_output);
    int actual_species = get_max_row(output);
    correct_predictions += (species_prediction == actual_species);
    
    // Backprop
    Matrix* delta_second_layer = get_delta_second_layer(calculated_output, output);
    neuralnet->weights[1] = add(neuralnet->weights[1], get_delta_second_weights(neuralnet, delta_second_layer, first_layer_output));
    neuralnet->biases[1] = add(neuralnet->biases[1], get_delta_second_biases(neuralnet, delta_second_layer));

    Matrix* delta_first_layer = get_delta_first_layer(neuralnet, first_layer_output, delta_second_layer);
    neuralnet->weight[0] = add(neuralnet->weight[0], get_delta_first_weights(neuralnet, delta_first_layer, input));
    neuralnet->biases[0] = add(neuralnet->biases[0], get_delta_first_biases(neuralnet, delta_first_layer));


    print_matrix(calculated_output);
    printf("Max row calculated_output = %d\n", species_prediction);
    printf("------------------\n");
    print_matrix(output);
    printf("Max row output = %d\n", actual_species);
    printf("Correct predictions = %d\n", correct_predictions);
    printf("Delta Second Layer: \n");
    print_matrix(delta_second_layer);
    */

    return 0;

}

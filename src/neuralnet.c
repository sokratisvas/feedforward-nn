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
            res->data[index_at(i, j, res)] = (matrix->data[index_at(i, j, matrix)] > 0) ? matrix->data[index_at(i, j, matrix)]:0;
            // res->data[index_at(i, j, res)] = tanh(matrix->data[index_at(i, j, matrix)]);
        }
    }
    return res;
}

Matrix* der_activate_matrix(Matrix* matrix) {
    Matrix* res = new_matrix(matrix->rows, matrix->columns);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            res->data[index_at(i, j, res)] = (matrix->data[index_at(i, j, matrix)] > 0) ? 1:0;
            // res->data[index_at(i, j, res)] = 1 - tanh(matrix->data[index_at(i, j, matrix)]) * tanh(matrix->data[index_at(i, j, matrix)]);
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
    return softmax(second_layer_output);
}

double means_squared_method(Matrix* calculated_output, Matrix* output) {
    assert(output->columns == 1 && calculated_output->columns == 1);
    assert(calculated_output->rows == output->rows);
    double error = 0;
    for (int i = 0; i < calculated_output->rows; i++) {
          error += (calculated_output->data[i] - output->data[i]) * (calculated_output->data[i] - output->data[i]);
    }
    return error / calculated_output->rows;
}

double cross_entropy_method(Matrix* calculated_output, Matrix* output) {
    assert(output->columns == 1 && calculated_output->columns == 1);
    assert(calculated_output->rows == output->rows);
    int max_row = get_max_row(output);
    return -log(calculated_output->data[max_row]);
}

Matrix* get_delta_second_layer(Matrix* calculated_output, Matrix* output) {
    return subtract(calculated_output, output); 
}

Matrix* get_delta_second_weights(NeuralNet* neuralnet, Matrix* delta_second_layer, Matrix* first_layer_output) {
    return multiply_by_scalar(multiply(delta_second_layer, transpose(first_layer_output)), -0.33 * neuralnet->learning_rate);
}

Matrix* get_delta_second_biases(NeuralNet* neuralnet, Matrix* delta_second_layer) {
    return multiply_by_scalar(delta_second_layer, -0.33 * neuralnet->learning_rate);
}

Matrix* get_delta_first_layer(NeuralNet* neuralnet, Matrix* first_layer_output, Matrix* delta_second_layer) {
    return multiply_elementwise(multiply(transpose(neuralnet->weights[1]), delta_second_layer), der_activate_matrix(first_layer_output));
}

Matrix* get_delta_first_weights(NeuralNet* neuralnet, Matrix* delta_first_layer, Matrix* input) {
    return multiply_by_scalar(multiply(delta_first_layer, transpose(input)), -0.33 * neuralnet->learning_rate);
}

Matrix* get_delta_first_biases(NeuralNet* neuralnet, Matrix* delta_first_layer) {
    return multiply_by_scalar(delta_first_layer, -0.33 * neuralnet->learning_rate);
}

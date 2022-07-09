#ifndef NEURALNET_H
#define NEURALNET_H

#define INPUT_LAYER_SIZE 4
#define HIDDEN_LAYER_SIZE 10
#define OUTPUT_LAYER_SIZE 3
#define LEARNING_RATE 0.01
#define EPOCHS 600

typedef struct {
    int* neurons_per_layer;
    Matrix** weights;
    Matrix** biases;
    double learning_rate;
} NeuralNet;

NeuralNet* new_neuralnet(int* neurons_per_layer, double learning_rate);
void delete_neuralnet(NeuralNet* neuralnet);
Matrix* activate_matrix(Matrix* matrix);
Matrix* der_activate_matrix(Matrix* matrix);
Matrix* softmax(Matrix* matrix);
Matrix* get_prefirst_layer_output(NeuralNet* neuralnet, Matrix* input); // Before applying the activation function on the 1st layer
Matrix* get_first_layer_output(NeuralNet* neuralnet, Matrix* input);
Matrix* get_second_layer_output(NeuralNet* neuralnet, Matrix* act_first_layer_output);
double means_squared_method(Matrix* calculated_output, Matrix* output);
double cross_entropy_method(Matrix* calculated_output, Matrix* output);
Matrix* get_delta_second_layer(Matrix* calculated_output, Matrix* output);
Matrix* get_delta_second_weights(NeuralNet* neuralnet, Matrix* delta_second_layer, Matrix* first_layer_output);
Matrix* get_delta_second_biases(NeuralNet* neuralnet, Matrix* delta_second_layer);
Matrix* get_delta_first_layer(NeuralNet* neuralnet, Matrix* first_layer_output, Matrix* delta_second_layer);
Matrix* get_delta_first_weights(NeuralNet* neuralnet, Matrix* delta_first_layer, Matrix* input);
Matrix* get_delta_first_biases(NeuralNet* neuralnet, Matrix* delta_first_layer);

#endif

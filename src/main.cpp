#include "Perceptron.hpp"
#include "InputNeuron.hpp"
// #include "SigmoidNeuron.hpp"

#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using decimal = float;

int main() {

    cout << "\nHello, Main!" << endl;

    int input_layer_size = 784;
    int first_layer_size = 16;
    vector <Perceptron<decimal>> input_layer(input_layer_size);
    vector <Perceptron<decimal>> first_layer(first_layer_size);

    return 0;
}
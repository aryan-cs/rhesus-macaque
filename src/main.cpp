#include "Perceptron.hpp"
#include "InputNeuron.hpp"
#include "SigmoidNeuron.hpp"
#include "NeuralNetwork.hpp"

#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using decimal = float;

int main() {

    cout << "\nHello, Main!" << endl;

    NeuralNetwork<Perceptron> nn({784, 16, 16, 10});

    cout << nn << endl;

    return 0;
}
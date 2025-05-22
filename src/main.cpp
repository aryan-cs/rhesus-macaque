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

    NeuralNetwork<SigmoidNeuron> nn({4, 3, 2});
    nn.Initialize();

    vector <decimal> inputs = {1.0, 1.0, 0.75, 1.0};
    nn.SetInputs(inputs);

    nn.FeedForward();

    cout << nn << endl;

    return 0;
}
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

    NeuralNetwork<Perceptron> nn({4, 3, 2});

    cout << nn << endl;

    nn.Initialize();
    cout << nn.GetInputLayer().size() << endl;

    for (vector <Perceptron> n : nn.GetHiddenLayers()) {
        cout << n[0].GetWeights().size() << endl;
    }

    return 0;
}
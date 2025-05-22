#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "InputNeuron.hpp"
#include "Perceptron.hpp"
#include "SigmoidNeuron.hpp"

#include <iostream>
#include <vector>

using namespace std;
using decimal = float;

template <typename T>
class NeuralNetwork {
public:
    NeuralNetwork () {};
    
    template <size_t N>
    NeuralNetwork(const int (&layers)[N]) {
        neural_network.emplace_back(layers[0]);
        
        for (int l = 1; l < N; ++l) {
            neural_network.emplace_back(layers[l]);
        }
    }
    
    ~NeuralNetwork () {};

    void HelloWorld () { cout << "Hello, Neural Network!" << endl; };

    friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<T>& nn) {
        
        for (vector<T> layer : nn.neural_network) { os << "[" << layer.size() << "] → "; }
        os << "[ ]";
        return os;
    }

    decimal Cost(vector <decimal> ideal_outputs) {
        decimal cost = 0.0;
        for (int o = 0; o < ideal_outputs.size(); ++o) {
            cost += ((neural_network[neural_network.size() - 1][o] - ideal_outputs[o])(neural_network[neural_network.size() - 1][o] - ideal_outputs[o]));
        }
        return cost;
    }
    
private:
    vector <vector <T>> neural_network;
};

#endif
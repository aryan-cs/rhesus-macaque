#ifndef hidden_layers_H
#define hidden_layers_H

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
        for (int i_p = 0; i_p < layers[0]; ++i_p) {
            input_layer.push_back(InputNeuron());
        }
        
        for (int l = 1; l < N; ++l) {
            hidden_layers.emplace_back(layers[l]);
        }
    }
    
    ~NeuralNetwork () {};

    void Initialize () {
        for (InputNeuron i_p : input_layer) {
            i_p.FeedNextLayer(hidden_layers[0]);
        }
        for (int l = 0; l < hidden_layers.size() - 1; ++l) {
            for (int n = 0; n < hidden_layers[l].size(); ++n) {
                hidden_layers[l][n].Initialize();
                hidden_layers[l][n].FeedNextLayer(hidden_layers[l + 1]);
            }
        }
    }

    void HelloWorld () { cout << "Hello, Neural Network!" << endl; };

    vector <InputNeuron> GetInputLayer () { return input_layer; }

    vector <vector <T>> GetHiddenLayers () { return hidden_layers; }

    friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<T>& nn) {
        os << "[" << nn.input_layer.size() << "] → "; 
        for (vector<T> layer : nn.hidden_layers) { os << "[" << layer.size() << "] → "; }
        os << "[ ]";
        return os;
    }

    decimal Cost(vector <decimal> ideal_outputs) {
        decimal cost = 0.0;
        for (int o = 0; o < ideal_outputs.size(); ++o) {
            cost += ((hidden_layers[hidden_layers.size() - 1][o] - ideal_outputs[o])(hidden_layers[hidden_layers.size() - 1][o] - ideal_outputs[o]));
        }
        return cost;
    }
    
private:
    vector <InputNeuron> input_layer;
    vector <vector <T>> hidden_layers;
};

#endif
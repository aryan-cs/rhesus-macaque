#ifndef hidden_layers_H
#define hidden_layers_H

#include "InputNeuron.hpp"

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

    void ClearNetwork () {
        for (InputNeuron& i_p : input_layer) {
            i_p.SetValue(0.0);
        }
        for (int l = 0; l < hidden_layers.size() - 1; ++l) {
            for (T& node : hidden_layers[l]) {
                node.ClearInputs();
            }
        }
    }

    void Initialize () {
        ClearNetwork();
        for (InputNeuron& i_p : input_layer) {
            i_p.ConnectTo(hidden_layers[0]);
        }
        for (int l = 0; l < hidden_layers.size(); ++l) {
            for (T& node : hidden_layers[l]) {
                if (l < hidden_layers.size() - 1) node.ConnectTo(hidden_layers[l + 1]);
                node.InitializeBias();
            }
        }
    }

    void HelloWorld () { cout << "Hello, Neural Network!" << endl; };

    void SetInputs (vector <decimal> inps) {
        for (int n = 0; n < input_layer.size(); ++n) {
            input_layer[n].SetValue(inps[n]);
        }
    }

    vector <InputNeuron>& GetInputLayer () { return input_layer; }

    vector <vector <T>>& GetHiddenLayers () { return hidden_layers; }

    decimal Cost(vector <decimal> ideal_outputs) {
        decimal cost = 0.0;
        int o = 0;
        for (; o < ideal_outputs.size(); ++o) {
            decimal error = hidden_layers.back()[o].GetOutput() - ideal_outputs[o];
            cost += error * error;
        }
        return cost / (2 * o);
    }



    void FeedForward () {
        for (int n = 0; n < input_layer.size(); ++n) {
            input_layer[n].FeedNextLayer(n, hidden_layers[0]);
        }
        for (int l = 0; l < hidden_layers.size() - 1; ++l) {
            for (int n = 0; n < hidden_layers[l].size(); ++n) {
                hidden_layers[l][n].CalculateActivation();
                hidden_layers[l][n].FeedNextLayer(n, hidden_layers[l + 1]);
            }
        }
        for (int n = 0; n < hidden_layers[hidden_layers.size() - 1].size(); ++n) {
            hidden_layers[hidden_layers.size() - 1][n].CalculateActivation();
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<T>& nn) {
        os << "[" << nn.input_layer.size() << "] → "; 
        for (vector<T> layer : nn.hidden_layers) { os << "[" << layer.size() << "] → "; }
        os << "[ ]";
        return os;
    }
    
private:
    vector <InputNeuron> input_layer;
    vector <vector <T>> hidden_layers;
};

#endif
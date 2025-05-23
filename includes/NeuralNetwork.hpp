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
    NeuralNetwork(const size_t (&layers)[N]) {
        for (size_t i_p = 0; i_p < layers[0]; ++i_p) {
            input_layer.push_back(InputNeuron());
        }
        
        for (size_t l = 1; l < N; ++l) {
            hidden_layers.emplace_back(layers[l]);
        }
    }
    
    ~NeuralNetwork () {};

    void ClearNetwork () {
        for (InputNeuron& i_p : input_layer) {
            i_p.SetValue(0.0);
        }
        for (size_t l = 0; l < hidden_layers.size() - 1; ++l) {
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
        for (size_t l = 0; l < hidden_layers.size(); ++l) {
            for (T& node : hidden_layers[l]) {
                if (l < hidden_layers.size() - 1) node.ConnectTo(hidden_layers[l + 1]);
                node.InitializeBias();
            }
        }
    }

    void HelloWorld () { cout << "Hello, Neural Network!" << endl; };

    void SetInputs (vector <decimal> inps) {
        for (size_t n = 0; n < input_layer.size(); ++n) {
            input_layer[n].SetValue(inps[n]);
        }
    }

    const vector <InputNeuron>& GetInputLayer () { return input_layer; }
    const vector <vector <T>>& GetHiddenLayers () { return hidden_layers; }

    const decimal Cost(vector <decimal> ideal_outputs) {
        decimal cost = 0.0;
        for (size_t o = 0; o < ideal_outputs.size(); ++o) {
            decimal error = hidden_layers.back()[o].GetOutput() - ideal_outputs[o];
            cost += error * error;
        }
        return cost / (2 * ideal_outputs.size());
    }

    void FeedForward () {
        for (size_t n = 0; n < input_layer.size(); ++n) {
            input_layer[n].FeedNextLayer(n, hidden_layers[0]);
        }
        for (size_t l = 0; l < hidden_layers.size() - 1; ++l) {
            for (size_t n = 0; n < hidden_layers[l].size(); ++n) {
                hidden_layers[l][n].CalculateActivation();
                hidden_layers[l][n].FeedNextLayer(n, hidden_layers[l + 1]);
            }
        }
        for (size_t n = 0; n < hidden_layers[hidden_layers.size() - 1].size(); ++n) {
            hidden_layers[hidden_layers.size() - 1][n].CalculateActivation();
        }
    }

    void BackPropagation(const vector<decimal>& target_outputs, decimal learning_rate) {
        size_t L = hidden_layers.size();
        vector<vector<decimal>> deltas(L);

        for (size_t i = 0; i < hidden_layers[L - 1].size(); ++i) {
            decimal y = hidden_layers[L - 1][i].GetOutput();
            decimal error = y - target_outputs[i];
            decimal derivative = y * (1 - y);
            deltas[L - 1].push_back(error * derivative);
        }

        for (int l = L - 2; l >= 0; --l) {
            auto& current_layer = hidden_layers[l];
            auto& next_layer = hidden_layers[l + 1];
            deltas[l].resize(current_layer.size());

            for (size_t i = 0; i < current_layer.size(); ++i) {
                decimal sum = 0;
                for (size_t j = 0; j < next_layer.size(); ++j) {
                    sum += next_layer[j].GetWeights()[i] * deltas[l + 1][j];
                }
                decimal y = current_layer[i].GetOutput();
                deltas[l][i] = sum * y * (1 - y);
            }
        }

        for (size_t l = 0; l < L; ++l) {
            for (size_t i = 0; i < hidden_layers[l].size(); ++i) {
                auto& neuron = hidden_layers[l][i];
                auto& weights = neuron.GetWeights();
                auto& inputs = neuron.GetInputs();
                decimal delta = deltas[l][i];

                for (size_t j = 0; j < weights.size(); ++j) {
                    neuron.UpdateWeight(j, learning_rate * delta * inputs[j]);
                }

                neuron.UpdateBias(learning_rate * delta);
            }
        }
    }


    const friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<T>& nn) {
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
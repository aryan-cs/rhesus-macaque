#ifndef SIGMOID_NEURON_H
#define SIGMOID_NEURON_H

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;
using decimal = float;

class SigmoidNeuron : public Perceptron {
public:
    SigmoidNeuron () { /* cout << "> Sigmoid Neuron created!" << endl; */ };
    ~SigmoidNeuron () { /* cout << "> Sigmoid Neuron destroyed!" << endl */ };

    void HelloWorld () { cout << "Hello, Sigmoid Neuron!" << endl; };

    vector <decimal> GetInputs () { return inputs; }
    decimal GetOutput () { return output; }

    decimal ActivationFunction () {
        // z = w.i + b
        decimal z = inner_product(weights.begin(), 
                                  weights.end(), 
                                  inputs.begin(), 
                                  bias);

        // Sigmoid Function: 1 / (1 + exp(-z))
        output = 1 / (1 + exp(-z));

        return output;
    }

    void FeedNextLayer (vector<SigmoidNeuron> next_layer) {
        for (SigmoidNeuron s : next_layer) {
            s.GetInputs().clear();
            s.GetInputs().push_back(output);
        }
    }
    
private:
    vector <decimal> inputs;
    vector <decimal> weights;
    decimal bias;
    decimal output;
};

#endif
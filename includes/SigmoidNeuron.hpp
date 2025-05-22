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

    void AddInput (decimal i) { inputs.push_back(i); }
    vector <decimal>& GetInputs () { return inputs; }
    void ClearInputs () { inputs.clear(); }

    void AddWeight(decimal w) { weights.push_back(w); }
    vector <decimal>& GetWeights () { return weights; }

    decimal& GetOutput () { return output; }

    void AddConnection () {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static normal_distribution<decimal> dist(/* mean = */ 0.0, /* stdev = */ 0.25);

        inputs.push_back(dist(gen));
        weights.push_back(dist(gen));
    }

    void InitializeBias () {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static normal_distribution<decimal> dist(/* mean = */ 0.0, /* stdev = */ 0.25);
        
        bias = dist(gen);
    }

    void ConnectTo (vector<SigmoidNeuron>& next_layer) {
        for (SigmoidNeuron& s : next_layer) {
            s.AddConnection();
        }
    }

    void FeedNextLayer (vector<SigmoidNeuron>& next_layer) {
        for (SigmoidNeuron& s : next_layer) {
            s.ClearInputs();
            s.AddInput(output);
        }
    }
    
private:
    vector <decimal> inputs;
    vector <decimal> weights;
    decimal bias;
    decimal output = 0.0;
};

#endif
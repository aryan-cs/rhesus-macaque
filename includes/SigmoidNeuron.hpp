#ifndef SIGMOID_NEURON_H
#define SIGMOID_NEURON_H

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;
using decimal = float;

class SigmoidNeuron {
public:
    SigmoidNeuron () { /* cout << "> Sigmoid Neuron created!" << endl; */ };
    ~SigmoidNeuron () { /* cout << "> Sigmoid Neuron destroyed!" << endl */ };

    void HelloWorld () { cout << "Hello, Sigmoid Neuron!" << endl; };

    void SetInput (decimal i, decimal v) { inputs[i] = v; }
    const vector <decimal>& GetInputs () { return inputs; }
    void ClearInputs () { inputs.clear(); }
    void AddWeight(decimal w) { weights.push_back(w); }
    const vector <decimal>& GetWeights () { return weights; }
    void UpdateWeight(int i, decimal delta) { weights[i] -= delta; }
    void UpdateBias(decimal delta) { bias -= delta; }

    const decimal& GetOutput () { return output; }

    void AddConnection () {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static normal_distribution<decimal> dist(/* mean = */ 0.0, /* stdev = */ 0.25);

        inputs.push_back(0);
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

    decimal CalculateActivation () {
        // z = w.i + b
        decimal z = inner_product(weights.begin(), 
                                  weights.end(), 
                                  inputs.begin(), 
                                  bias);

        // Sigmoid Function: 1 / (1 + exp(-z))
        output = 1 / (1 + exp(-z));

        return output;
    }

    void FeedNextLayer (int i, vector<SigmoidNeuron>& next_layer) {
        for (SigmoidNeuron& s : next_layer) {
            s.SetInput(i, output);
        }
    }
    
private:
    vector <decimal> inputs;
    vector <decimal> weights;
    decimal bias;
    decimal output = 0.0;
};

#endif
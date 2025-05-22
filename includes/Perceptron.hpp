#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using decimal = float;

class Perceptron {
public:
    Perceptron () {};
    ~Perceptron () {};

    void HelloWorld () { cout << "Hello, Perceptron!" << endl; };

    void Initialize () {
        srand(time(0));
        bias = ((decimal) rand()) / RAND_MAX;
        for (decimal w : weights) {
            w = ((decimal) rand()) / RAND_MAX;
        }
    }

    vector <decimal> GetInputs () { return inputs; }
    void ClearInputs () { inputs.clear(); }
    vector <decimal> GetWeights () { return weights; }
    bool GetOutput () { return output; }
    void InitializeInput () { inputs.push_back(0); }
    void InitializeWeight () { weights.push_back(0); }

    bool ActivationFunction () {
        // w.i + b
        decimal weighted_sum = inner_product(weights.begin(), 
                                             weights.end(), 
                                             inputs.begin(), 
                                             bias);

        // Heaviside Step Function: 1 iff w.i + b > 0
        output = (weighted_sum > 0);

        return output;
    }

    void FeedNextLayer (vector<Perceptron> next_layer) {
        for (Perceptron p : next_layer) {
            p.GetInputs().clear();
            p.GetInputs().push_back(output);
        }
    }
    
private:
    vector <decimal> inputs;
    vector <decimal> weights;
    decimal bias;
    bool output;
};

#endif
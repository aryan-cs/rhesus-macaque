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

    vector <decimal> GetInputs () { return inputs; }
    bool GetOutput () { return output; }

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
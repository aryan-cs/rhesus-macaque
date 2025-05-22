#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <numeric>
#include <random>

using namespace std;
using decimal = float;

class Perceptron {
public:
    Perceptron () {};
    ~Perceptron () {};

    void HelloWorld () { cout << "Hello, Perceptron!" << endl; };

    void AddInput (decimal i) { inputs.push_back(i); }
    vector <decimal>& GetInputs () { return inputs; }
    void ClearInputs () { inputs.clear(); }

    void AddWeight(decimal w) { weights.push_back(w); }
    vector <decimal>& GetWeights () { return weights; }

    bool& GetOutput () { return output; }

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


    void ConnectTo (vector<Perceptron>& next_layer) {
        for (Perceptron& p : next_layer) {
            p.AddConnection();
        }
    }

    void FeedNextLayer (vector<Perceptron>& next_layer) {
        for (Perceptron& p : next_layer) {
            p.ClearInputs();
            p.AddInput(output);
        }
    }
    
private:
    vector <decimal> inputs;
    vector <decimal> weights;
    decimal bias;
    bool output = 0;
};

#endif
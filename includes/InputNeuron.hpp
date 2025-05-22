#ifndef INPUT_NEURON_H
#define INPUT_NEURON_H


#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using decimal = float;

class InputNeuron {
public:
    InputNeuron() : value(decimal {}) {};
    InputNeuron (decimal v) : value(v) { };
    ~InputNeuron () {};

    void HelloWorld () { cout << "Hello, Input Neuron!" << endl; };

    decimal GetValue () { return value; }

    void SetValue (const decimal& v) { value = v; }
    
    template <typename T>
    void FeedNextLayer (vector<T> next_layer) {
        for (T nueron : next_layer) {
            nueron.GetInputs().push_back(value);
        }
    }

private:
    decimal value;
};

#endif
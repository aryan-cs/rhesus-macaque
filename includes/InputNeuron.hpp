#ifndef INPUT_NEURON_H
#define INPUT_NEURON_H


#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using decimal = float;

template <typename T>
class InputNeuron {
public:
    InputNeuron() : value(T{}) { /* cout << "> Input Neuron created!" << endl; */ };
    InputNeuron (T v) : value(v) { };
    ~InputNeuron () { /* cout << "> Input Neuron destroyed!" << endl */ };

    void HelloWorld () { cout << "Hello, Input Neuron!" << endl; };

    T GetValue () { return value; }

    void SetValue (const decimal& v) { value = v; }
    
    void FeedNextLayer (vector<T> next_layer) {
        for (T nueron : next_layer) {
            nueron.GetInputs().push_back(value);
        }
    }

private:
    T value;
};

#endif
#ifndef INPUT_NEURON_H
#define INPUT_NEURON_H


#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using decimal = float;

class InputNeuron {
public:
    InputNeuron() {};
    ~InputNeuron () {};

    void HelloWorld () { cout << "Hello, Input Neuron!" << endl; };

    decimal GetValue () { return value; }

    void SetValue (const decimal& v) { value = v; }
    
    template <typename T>
    void InitializeNextLayer (vector<T>& next_layer) {
        for (T& nueron : next_layer) {
            // nueron.ClearInputs();
            nueron.InitializeInput();
            nueron.InitializeWeight();
        }
    }

    template <typename T>
    void FeedNextLayer (vector<T>& next_layer) {
        for (T& nueron : next_layer) {
            // continue here
            // think about indexing
        }
    }

private:
    decimal value;
};

#endif
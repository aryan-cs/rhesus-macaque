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
    void ConnectTo (vector<T>& next_layer) {
        for (T& nueron : next_layer) {
            nueron.AddConnection();
        }
    }

    template <typename T>
    void FeedNextLayer (int i, vector<T>& next_layer) {
        for (T& n : next_layer) {
            n.SetInput(i, value);
        }
    }

private:
    decimal value;
};

#endif
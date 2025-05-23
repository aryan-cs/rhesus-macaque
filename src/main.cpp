#include "Perceptron.hpp"
#include "InputNeuron.hpp"
#include "SigmoidNeuron.hpp"
#include "NeuralNetwork.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;
using decimal = float;

struct TestData {
    vector<decimal> input;
    vector<decimal> output;
};

int main() {
    cout << "\nHello, Main!\n" << endl;

    NeuralNetwork<SigmoidNeuron> nn({4, 3, 2});
    nn.Initialize();

    cout << nn;

    vector<TestData> data = {
        {{1, 1, 1, 1}, {1, 1}},
        {{1, 1, 0, 0}, {1, 0}},
        {{0, 0, 1, 1}, {0, 1}},

        {{1, 1, 1, 0}, {1, 0.5}},
        {{0, 1, 1, 1}, {0.5, 1}},

        {{1, 0, 1, 0}, {0.5, 0.5}},
        {{1, 0, 0, 1}, {0.5, 0.5}},
        {{0, 1, 0, 1}, {0.5, 0.5}},
        {{0, 1, 1, 0}, {0.5, 0.5}},

        {{0, 1, 0, 0}, {0.5, 0}},
        {{0, 0, 1, 0}, {0, 0.5}},

        {{0, 0, 0, 0}, {0, 0}},
    };

    decimal total_cost = 0;
    for (auto& d : data) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        total_cost += nn.Cost(d.output);
    }

    cout << "\nInitial Average Cost: " << total_cost / data.size() << endl;

    cout << "\nTraining...\n" << endl;
    int epochs = 5000;
    decimal lr = 0.5;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (auto& d : data) {
            nn.SetInputs(d.input);
            nn.FeedForward();
            nn.BackPropagation(d.output, lr);
        }

        if (epoch % 1000 == 0) {
            decimal epoch_cost = 0;
            for (auto& d : data) {
                nn.SetInputs(d.input);
                nn.FeedForward();
                epoch_cost += nn.Cost(d.output);
            }
            cout << "Epoch " << epoch << " | Average Cost: " << epoch_cost / data.size() << endl;
        }
    }

    cout << "\nPost-train Results:\n";
    const float epsilon = 1e-7;

    for (auto& d : data) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        auto preds = nn.GetHiddenLayers().back();

        cout << "Input: [";
        for (auto v : d.input) cout << v << ", ";
        cout << "]\nPredicted: [";
        for (auto& n : preds) cout << round(n.GetOutput() * 100) / 100.0 << ", ";
        cout << "]\nTarget: [";
        for (auto v : d.output) cout << v << ", ";

        cout << "]\n" << endl;
    }

    return 0;
}

#include "Perceptron.hpp"
#include "InputNeuron.hpp"
#include "SigmoidNeuron.hpp"
#include "NeuralNetwork.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>

using namespace std;
using decimal = float;

const decimal PRECISION = 1000.0;
const decimal ERROR_THRESHOLD = 1;
int EPOCHS = 5000;
decimal LEARNING_RATE = 0.5;

struct TestData {
    vector<decimal> input;
    vector<decimal> output;
};

decimal average_of (decimal a, decimal b) { return (a + b) / 2.0f; }

decimal with_precision (decimal n) { return round(n * PRECISION) / PRECISION; }

decimal percentage_error (const vector<decimal>& preds, const vector<decimal>& targets) {
    decimal error = 0.0;

    for (size_t n = 0; n < preds.size(); ++n) {
        decimal term = abs(targets[n] - preds[n]);
        if (targets[n] != 0) { term /= targets[n]; }
        error += term;
    }

    return with_precision(error);
}


int main() {
    cout << "\nHello, Main!\n" << endl;

    NeuralNetwork<SigmoidNeuron> nn({4, 3, 2});
    nn.Initialize();

    cout << "Network Architecture: " << endl;
    cout << nn;

    // Test Application: One-dimensional Array Quantization
    // Ex: [1, 1, 0, 0] -> [1, 0]
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

    cout << "\n\nInitial Average Cost: " << total_cost / data.size() << endl;

    cout << "\nTraining...\n" << endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (auto& d : data) {
            nn.SetInputs(d.input);
            nn.FeedForward();
            nn.BackPropagation(d.output, LEARNING_RATE);
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

    cout << "\n--------------------------------------------------------------------------------" << endl;
    cout << "\nPost-train Results:\n\n";

    for (auto& d : data) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        auto preds = nn.GetHiddenLayers().back();

        vector<decimal> preds_vec;
        for (auto& n : preds) preds_vec.push_back(n.GetOutput());

        decimal error = percentage_error(preds_vec, d.output);

        cout << "[";
        for (auto v : d.input) cout << with_precision(v) << ", ";
        cout << "] -> [";
        for (auto& n : preds) cout << with_precision(n.GetOutput()) << ", ";
        cout << "] ~> [";
        for (auto v : d.output) cout << with_precision(v) << ", ";
        cout << "]\nError: " << with_precision(error) << "%";
        cout << "\n" << endl;
    }

    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "\nValidation Results:\n";

    vector<TestData> validation_data;

    std::mt19937 rng(0);
    std::uniform_real_distribution<decimal> dist(0.0, 3.0);

    const int val_samples = 10;
    for (int i = 0; i < val_samples; ++i) {
        vector<decimal> input(4);
        for (auto& v : input) { v = dist(rng); }
        vector<decimal> output = {average_of(input[0], input[1]), average_of(input[2], input[3])};
        validation_data.push_back({input, output});
    }

    for (auto& d : validation_data) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        auto preds = nn.GetHiddenLayers().back();

        vector<decimal> preds_vec;
        for (auto& n : preds) preds_vec.push_back(n.GetOutput());

        decimal error = percentage_error(preds_vec, d.output);

        cout << "[";
        for (auto v : d.input) cout << with_precision(v) << ", ";
        cout << "] -> [";
        for (auto& n : preds) cout << with_precision(n.GetOutput()) << ", ";
        cout << "] ~> [";
        for (auto v : d.output) cout << with_precision(v) << ", ";
        cout << "]\nError: " << with_precision(error) << "%";
        cout << "\n" << endl;
    }

    auto compute_accuracy = [&](const vector<TestData>& dataset) {
        int correct_count = 0;
        for (auto& d : dataset) {
            nn.SetInputs(d.input);
            nn.FeedForward();
            auto preds = nn.GetHiddenLayers().back();

            vector<decimal> preds_vec;
            for (auto& n : preds) preds_vec.push_back(n.GetOutput());

            decimal error = percentage_error(preds_vec, d.output);
            if (error <= ERROR_THRESHOLD) { ++correct_count; }
        }

        return (decimal(correct_count) / dataset.size()) * 100.0f;
    };

    cout << "--------------------------------------------------------------------------------" << endl;

    decimal train_accuracy = compute_accuracy(data);
    cout << "\nTraining Data Accuracy   (error ≤ " << ERROR_THRESHOLD << "%): " 
        << with_precision(train_accuracy) << "%\n";

    decimal val_accuracy = compute_accuracy(validation_data);
    cout << "Validation Data Accuracy (error ≤ " << ERROR_THRESHOLD << "%): " 
        << with_precision(val_accuracy) << "%\n";

    return 0;
}

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

const string section = "---------------------------------------------";

// Network Architecture
const int INPUT_SIZE = 4;
const int OUTPUT_SIZE = 2;
const int EPOCHS = 10000; // Edit this!
const decimal LEARNING_RATE = 0.5; // Edit this!

// Designing Training Data
const int DATASET_SIZE = 1000; // Edit this!
const decimal ALLOCATED_TRAINING = 0.75;
const decimal TRAINING_BOUND_LOWER = 0.0;
const decimal TRAINING_BOUND_UPPER = 1.0;

// Designing Validation Data
const decimal ALLOCATED_VALIDATION = 1 - ALLOCATED_TRAINING;
const decimal VALIDATION_BOUND_LOWER = 0.0;
const decimal VALIDATION_BOUND_UPPER = 1.0;

// Benchmark Metrics
const decimal PRECISION = 10000.0;
const decimal ERROR_THRESHOLD = 0.1; // Edit this!

struct Data {
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

vector <Data> DataSetBuilder (int DATASET_SIZE, 
                              decimal DATA_LOWER, 
                              decimal DATA_UPPER) {

    std::mt19937 rng(0);
    std::uniform_real_distribution<decimal> dist(DATA_LOWER, DATA_UPPER);

    vector <Data> dataset;

    for (int d = 0; d < DATASET_SIZE; ++d) {

        vector<decimal> input(4);
        for (auto& v : input) { v = dist(rng); }

        vector<decimal> output = {average_of(input[0], input[1]), average_of(input[2], input[3])};
        dataset.push_back({input, output});

    }

    return dataset;

}


int main() {
    cout << "\nHello, Main!\n" << endl;

    NeuralNetwork<SigmoidNeuron> nn({INPUT_SIZE, 3, OUTPUT_SIZE});
    nn.Initialize();

    cout << "Network Architecture: " << endl;
    cout << nn;

    vector <Data> training_dataset = DataSetBuilder(round(DATASET_SIZE * ALLOCATED_TRAINING), 
                                                    TRAINING_BOUND_LOWER, 
                                                    TRAINING_BOUND_UPPER);

    cout << "\n\nTraining Dataset Size: " << training_dataset.size() << endl;

    vector <Data> validation_dataset = DataSetBuilder(round(DATASET_SIZE * ALLOCATED_VALIDATION), 
                                                      VALIDATION_BOUND_LOWER, 
                                                      VALIDATION_BOUND_UPPER);

    cout << "Validation Dataset Size: " << validation_dataset.size() << endl;

    decimal total_cost = 0;
    for (auto& d : training_dataset) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        total_cost += nn.Cost(d.output);
    }

    cout << "Initial Average Cost: " << total_cost / training_dataset.size() << endl;

    cout << "\nTraining...\n" << endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (auto& d : training_dataset) {
            nn.SetInputs(d.input);
            nn.FeedForward();
            nn.BackPropagation(d.output, LEARNING_RATE);
        }



        if (epoch % 1000 == 0) {
            decimal epoch_cost = 0;
            for (auto& d : training_dataset) {
                nn.SetInputs(d.input);
                nn.FeedForward();
                epoch_cost += nn.Cost(d.output);
            }
            cout << "Epoch " << epoch << " | Average Cost: " << epoch_cost / training_dataset.size() << endl;
        }
    }

    // cout << "\n" << section << endl;
    // cout << "\nPost-train Results:\n\n";

    for (auto& d : training_dataset) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        auto preds = nn.GetHiddenLayers().back();

        vector<decimal> preds_vec;
        for (auto& n : preds) preds_vec.push_back(n.GetOutput());

        decimal error = percentage_error(preds_vec, d.output);

        // cout << "[";
        // for (auto v : d.input) cout << with_precision(v) << ", ";
        // cout << "] -> [";
        // for (auto& n : preds) cout << with_precision(n.GetOutput()) << ", ";
        // cout << "] ~> [";
        // for (auto v : d.output) cout << with_precision(v) << ", ";
        // cout << "]\nError: " << with_precision(error) << "%";
        // cout << "\n" << endl;
    }

    // cout << section << endl;
    // cout << "\nValidation Results:\n";

    for (auto& d : validation_dataset) {
        nn.SetInputs(d.input);
        nn.FeedForward();
        auto preds = nn.GetHiddenLayers().back();

        vector<decimal> preds_vec;
        for (auto& n : preds) preds_vec.push_back(n.GetOutput());

        decimal error = percentage_error(preds_vec, d.output);

        // cout << "[";
        // for (auto v : d.input) cout << with_precision(v) << ", ";
        // cout << "] -> [";
        // for (auto& n : preds) cout << with_precision(n.GetOutput()) << ", ";
        // cout << "] ~> [";
        // for (auto v : d.output) cout << with_precision(v) << ", ";
        // cout << "]\nError: " << with_precision(error) << "%";
        // cout << "\n" << endl;
    }

    auto compute_accuracy = [&](const vector<Data>& dataset) {
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

    cout << section << endl;

    decimal train_accuracy = compute_accuracy(training_dataset);
    cout << "\nTraining Data Accuracy   (error ≤ " << ERROR_THRESHOLD << "%): " 
        << with_precision(train_accuracy) << "%\n";

    decimal val_accuracy = compute_accuracy(validation_dataset);
    cout << "Validation Data Accuracy (error ≤ " << ERROR_THRESHOLD << "%): " 
        << with_precision(val_accuracy) << "%\n";

    return 0;
}

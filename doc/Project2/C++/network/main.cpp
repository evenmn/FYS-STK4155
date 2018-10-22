#include <iostream>
#include <armadillo>
#include <tools.h>
#include <networks.h>
#include <multilayer.h>
#include <recall.h>
#include <ctime>
#include <fstream>
#include <string>

using namespace std;
using namespace arma;

int main() {

    // Declare global matrices
    Mat <double> X;                            // Input data
    vec t;                                     // Targets

    // --- Multilayer ---
    double eta = 0.01;                          //Learning rate
    int T = 100;                              //Number of iterations
    int h[] = {5};                    //2 hidden layer where each has 4 nodes
    int size_h = sizeof(h)/sizeof(*h);

    // Load data. Data around Tc is excluded, and the data is shuffled
    X.load("../../data/Ising2DFM_reSample_L40tcexcluded_shuffled.dat");

    X = shuffle(X);

    t = X.col(1600);                            // Split data matrix into t
    X.shed_col(1600);                           // and X

    int N = size(X)[0];           // Number of rows (sets)
    int N_train = int(N*4/5.);    // Use 80% of set for training

    mat X_train = X.head_rows(N_train);         // Training data sets
    mat X_test = X.tail_rows(N - N_train);      // Test data set
    vec t_train = t.head(N_train);              // Training targets
    vec t_test = t.tail(N - N_train);           // Test targets


    field <mat> W_field(size_h+1, 1);           // Field containing all W-weights
    field <vec> b_field(size_h+1, 1);           // Field containing all b-weights
    mat out = zeros <mat> (size(t_test));       // Output matrix
    mat out2 = zeros <mat> (size(t_train));     // Output matrix

    clock_t start_multi = clock();
    multi(X_train, t_train, W_field, b_field, h, size_h, T, eta);
    clock_t end_multi = clock();

    //cout << W_field << endl;

    cout << "--- Multilayer ---" << endl;
    cout << "Test set" << endl;
    recall_multi(X_test, W_field, b_field, out); // Recall
    cout << trans(out.head_rows(20)) << endl;
    cout << trans(t_test.head(20)) << endl;
    cout << "Success rate: " << 1-sum(abs(round(t_test-out)))/(N-N_train) << endl;

    cout << "\nTraining set" << endl;
    recall_multi(X_train, W_field, b_field, out2); // Recall
    cout << trans(out2.head_rows(20)) << endl;
    cout << trans(t_train.head(20)) << endl;
    cout << "Success rate: " << 1-sum(abs(round(t_train-out2)))/N_train << endl;

    cout << "\nCPU-time training: " << 1.0*(end_multi - start_multi)/CLOCKS_PER_SEC << '\n' << endl;

    return 0;
}

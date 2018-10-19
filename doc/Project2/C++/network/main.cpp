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
    double eta = 0.1;                          //Learning rate
    int T = 500;                              //Number of iterations
    int h[] = {10, 10};                    //2 hidden layer where each has 4 nodes
    int size_h = sizeof(h)/sizeof(*h);

    // Load data. Data around Tc is excluded, and the data is shuffled
    X.load("../../data/Ising2DFM_reSample_L40tcexcluded_shuffled.csv", csv_ascii);

    t = X.col(1600);                            // Split data matrix into t
    X.shed_col(1600);                           // and X

    unsigned int N = size(X)[0];                // Number of rows (sets)
    unsigned int N_train = N*4/5.;              // Use 80% of set for training


    mat X_train = X.head_rows(N_train);         // Training data sets
    mat X_test = X.tail_rows(N - N_train);      // Test data set
    vec t_train = t.head(N_train);              // Training targets
    vec t_test = t.tail(N - N_train);           // Test targets
    field <mat> W_field(size_h+1, 1);           // Field containing all W-weights
    field <vec> b_field(size_h+1, 1);           // Field containing all b-weights
    mat out = zeros <mat> (size(t_test));       // Output matrix

    clock_t start_multi = clock();
    multi(X_train, t_train, W_field, b_field, h, size_h, T, eta);
    recall_multi(X_test, W_field, b_field, out); // Recall
    clock_t end_multi = clock();

    cout << "--- Multilayer ---" << endl;
    cout << trans(out.head_rows(20)) << endl;
    cout << trans(t_test.head(20)) << endl;
    cout << "Absolute error: " << sum(abs(round(t_test-out))) << endl;
    cout << "CPU-time: " << 1.0*(end_multi - start_multi)/CLOCKS_PER_SEC << '\n' << endl;

    return 0;
}

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

    // Constants
    double eta = 0.2;       //Learning rate

    // Declare global matrices
    Mat <double> X;         //Input arrays
    vec t;         //Output arrays

    // --- Multilayer ---
    int T = 10000;                 //Number of iterations
    int h[] = {4, 4};     //3 hidden layer where each has 10 nodes
    int size_h = sizeof(h)/sizeof(*h);

    // Load data
    X.load("../../data/Ising2DFM_reSample_L40.csv", csv_ascii);

    shuffle(X);                             // Shuffle X

    t = X.col(1600);                        // Split data matrix into t
    X.shed_col(1600);                       // and X

    int N = size(X)[0]/1000;                     // Number of rows (sets)
    int N_train = N*4/5.;                   // Use 80% of set for training



    mat X_train = X.head_rows(N_train);
    mat X_test = X.tail_rows(N - N_train);
    vec t_train = t.head(N_train);
    vec t_test = t.tail(N - N_train);


    /*
    // Set value to matrices
    X.resize(4, 2);
    X(0,0) = 0; X(0,1) = 0;
    X(1,0) = 0; X(1,1) = 1;
    X(2,0) = 1; X(2,1) = 0;
    X(3,0) = 1; X(3,1) = 1;

    t.resize(4, 1);
    t(0,0) = 0; t(1,0) = 1; t(2,0) = 1; t(3,0) = 0;
    //t(0,1) = 0; t(1,1) = 1; t(2,1) = 1; t(3,1) = 0;
    */

    field <mat> W_field(size_h+1, 1);
    field <vec> b_field(size_h+1, 1);
    mat out_3 = zeros <mat> (size(t_test));

    clock_t start_multi = clock();
    multi(X_train, t_train, W_field, b_field, h, size_h, T, eta);
    recall_multi(X_test, W_field, b_field, out_3); // Recall
    clock_t end_multi = clock();

    cout << "--- Multilayer ---" << endl;
    cout << trans(out_3.head_rows(50)) << endl;
    cout << trans(t_test.head(50)) << endl;
    cout << "MSE: " << sum(abs(t_test-out_3)) << endl;
    cout << "CPU-time: " << 1.0*(end_multi - start_multi)/CLOCKS_PER_SEC << '\n' << endl;

    return 0;
}

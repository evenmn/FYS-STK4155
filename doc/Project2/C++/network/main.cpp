#include <iostream>
#include <armadillo>
#include <tools.h>
#include <networks.h>
#include <multilayer.h>
#include <recall.h>
#include <ctime>

using namespace std;
using namespace arma;

int main() {

    // Constants
    double eta = 0.2;       //Learning rate
    int T, H;

    // Declare global matrices
    Mat <double> X;         //Input arrays
    Mat <double> t;         //Output arrays


    /*
    // --- Linear --- (OR gate)
    T = 500;               //Number of iterations

    Mat <double> W;        //Declare weight matrix
    vec b;                 //Declare vector for bias weights

    // Set value to input and output matrices
    X.resize(4, 2);
    X(0,0) = 0; X(0,1) = 0;
    X(1,0) = 0; X(1,1) = 1;
    X(2,0) = 1; X(2,1) = 0;
    X(3,0) = 1; X(3,1) = 1;

    t.resize(4, 1);
    t(0,0) = 0; t(1,0) = 1; t(2,0) = 1; t(3,0) = 1;
    //t(0,1) = 0; t(1,1) = 1; t(2,1) = 1; t(3,1) = 1;

    mat out_1 = zeros <mat> (size(t));

    // Training the network
    clock_t start_lin = clock();
    linear(X, t, W, b, T, eta);

    // Recalling (Using the outcomes of the training to get useful results)
    recall_linear(X, W, b, out_1);
    clock_t end_lin = clock();

    cout << "--- Linear ---" << endl;
    cout << round(out_1);
    cout << "CPU-time: " << 1.0*(end_lin - start_lin)/CLOCKS_PER_SEC << '\n' << endl;



    // --- Non-linear --- (XOR gate)
    T = 10000;            //Number of iterations
    H = 10;               //Number of hidden nodes

    Mat <double> W1;
    Mat <double> W2;
    vec b1;
    vec b2;

    // Set value to matrices
    X.resize(4, 2);
    X(0,0) = 0; X(0,1) = 0;
    X(1,0) = 0; X(1,1) = 1;
    X(2,0) = 1; X(2,1) = 0;
    X(3,0) = 1; X(3,1) = 1;

    t.resize(4, 1);
    t(0,0) = 0; t(1,0) = 1; t(2,0) = 1; t(3,0) = 0;
    //t(0,1) = 0; t(1,1) = 1; t(2,1) = 1; t(3,1) = 0;

    mat out_2 = zeros <mat> (size(t));

    clock_t start_nonlin = clock();
    nonlinear(X, t, W1, W2, b1, b2, T, H, eta);

    // Recall
    recall_nonlinear(X, W1, W2, b1, b2, out_2);
    clock_t end_nonlin = clock();

    cout << "--- Non-linear ---" << endl;
    cout << round(out_2);
    cout << "CPU-time: " << 1.0*(end_nonlin - start_nonlin)/CLOCKS_PER_SEC << '\n' << endl;


    */
    // --- Multilayer ---
    T = 1000000;                 //Number of iterations
    int h[] = {15, 15};     //3 hidden layer where each has 10 nodes
    int size_h = sizeof(h)/sizeof(*h);

    field <mat> W_field(size_h+1, 1);
    field <vec> b_field(size_h+1, 1);
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

    t.resize(9,1);
    t(0,0) = 3.91499;
    t(1,0) = 2.16447;
    t(2,0) = 1.70161;
    t(3,0) = 1.53939;
    t(4,0) = 1.50000;
    t(5,0) = 1.52581;
    t(6,0) = 1.58801;
    t(7,0) = 1.66771;
    t(8,0) = 1.77021;

    X.resize(9,1);
    X(0,0) = 0.1;
    X(1,0) = 0.2;
    X(2,0) = 0.3;
    X(3,0) = 0.4;
    X(4,0) = 0.5;
    X(5,0) = 0.6;
    X(6,0) = 0.7;
    X(7,0) = 0.8;
    X(8,0) = 0.9;

    Mat <double> X_interpol;
    X_interpol.resize(8,1);
    X_interpol(0,0) = 0.15;
    X_interpol(1,0) = 0.25;
    X_interpol(2,0) = 0.35;
    X_interpol(3,0) = 0.45;
    X_interpol(4,0) = 0.55;
    X_interpol(5,0) = 0.65;
    X_interpol(6,0) = 0.75;
    X_interpol(7,0) = 0.85;

    mat out_3 = zeros <mat> (size(X_interpol));

    clock_t start_multi = clock();
    multi(X, t, W_field, b_field, h, size_h, T, eta);
    recall_multi(X_interpol, W_field, b_field, out_3); // Recall
    clock_t end_multi = clock();

    cout << "--- Multilayer ---" << endl;
    cout << out_3 << endl;
    cout << "CPU-time: " << 1.0*(end_multi - start_multi)/CLOCKS_PER_SEC << '\n' << endl;
    return 0;
}

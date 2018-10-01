#include <iostream>
#include <armadillo>
#include <tools.h>

using namespace arma;

void linear(mat X, mat t, mat &W, vec &b, int T, double eta) {

    int I = size(X)[1];
    int O = size(t)[1];
    int M = size(X)[0];
    int N = size(t)[0];

    if(N != M){
        cout << "Input and output array do not have the same length, rejecting" << endl;
        exit(0);
    }

    W = 2*randu(I, O) - 1;
    b = 2*randu(O) - 1;

    // Declare vectors
    vec out = zeros <vec> (O);
    vec out_der = zeros <vec> (O);

    for(int iter = 0; iter<T; iter++) {
        for(int i = 0; i<M; i++) {
            // Forward Propagation
            vec net = trans(X.row(i) * W) + b;
            sigmoid(out, net);

            // Backward propagation
            sig_der(out_der, out);
            vec deltao = out_der % (out - trans(t.row(i)));

            // Update weights
            W = W - eta * trans(deltao * X.row(i));
            b = b - eta * deltao;
        }
    }
}


void nonlinear(mat X, mat t, mat &W1, mat &W2, vec &b1, vec &b2, int T, int H, double eta) {

    int I = size(X)[1];
    int O = size(t)[1];
    int M = size(X)[0];
    int N = size(t)[0];

    if(N != M){
        cout << "Input and output array do not have the same length, rejecting" << endl;
        exit(0);
    }

    // Weights
    W1 = 2*randu(I, H) - 1;
    W2 = 2*randu(H, O) - 1;

    b1 = 2*randu (H) - 1;
    b2 = 2*randu (O) - 1;

    // Declare vectors
    vec out_h = zeros <vec> (H);
    vec out_o = zeros <vec> (O);
    vec out_h_der = zeros <vec> (H);
    vec out_o_der = zeros <vec> (O);

    for(int iter=0; iter<T; iter++) {
        for(int i=0; i<M; i++) {
            // Forward propagation
            vec net_h = trans(X.row(i) * W1) + b1;
            sigmoid(out_h, net_h);

            vec net_o = trans(W2) * out_h + b2;
            sigmoid(out_o, net_o);

            // Backward propagation
            sig_der(out_o_der, out_o);
            vec deltao = out_o_der % (out_o - trans(t.row(i)));

            sig_der(out_h_der, out_h);
            vec deltah = out_h_der % (W2 * deltao);

            // Update weights
            b1 = b1 - eta * deltah;
            b2 = b2 - eta * deltao;
            W1 = W1 - eta * trans(deltah * X.row(i));
            W2 = W2 - eta * out_h * trans(deltao);
        }
    }
}

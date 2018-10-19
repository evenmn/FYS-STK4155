#include <iostream>
#include <armadillo>
#include <tools.h>
#include <random>

using namespace std;
using namespace arma;

void multi(mat X, mat t, field <mat> &W, field <vec> &b, int h[], int H, int T, double eta) {

    int I = size(X)[1];
    int O = size(t)[1];
    int M = size(X)[0];
    int N = size(t)[0];

    if(N != M){
        cout << "Input and output array do not have the same length, rejecting" << endl;
        exit(0);
    }


    W(0,0) = 2*randu(I, h[0]) - 1;
    b(0,0) = 2*randu(h[0]) - 1;
    W(H,0) = 2*randu(h[H-1], O) - 1;
    b(H,0) = 2*randu(O) - 1;

    for(int i=0; i<H-1; i++){
        W(i+1,0) = 2*randu(h[i], h[i+1]) - 1;
        b(i+1,0) = 2*randu(h[i+1]) - 1;
    }

    // Declare armadillo objects
    vec out_der = zeros <vec> (O);
    field <vec> out(H+2);
    field <vec> deltah(H+1);

    for(int iter=0; iter<T; iter++) {
        cout << iter+1 << '/' << T << endl;
        for(int i=0; i<M; i++) {

            // Forward propagation
            out(0,0) = trans(X.row(i));
            for(int j=0; j<H+1; j++) {
                vec net = trans(W(j,0)) * out(j,0) + b(j,0);
                sigmoid(out(j+1,0), net);
            }

            // Backward propagation
            sig_der(out_der, out(H+1,0));
            vec deltao = out_der % (out(H+1,0) - trans(t.row(i)));

            deltah(0,0) = deltao;
            for(int j=0; j<H; j++) {
                vec out_der_1 = zeros <vec> (h[H-1-j]);
                sig_der(out_der_1, out(H-j,0));
                deltah(j+1,0) = (W(H-j,0) * deltah(j,0)) % out_der_1;
            }

            for(int j=0; j<H+1; j++) {
                W(j,0) = W(j,0) - eta * out(j,0) * trans(deltah(H-j,0));
                b(j,0) = b(j,0) - eta * deltah(H-j,0);
            }
        }
    }
}

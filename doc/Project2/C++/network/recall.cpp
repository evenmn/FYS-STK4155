#include <iostream>
#include <armadillo>
#include <tools.h>

using namespace arma;

void recall_linear(mat X, mat W, vec b, mat &out){
    for(int i = 0; i<size(X)[0]; i++) {
        vec net = trans(X.row(i) * W) + b;
        vec out_Xi = zeros <vec> (size(net)[1]);
        sigmoid(out_Xi, net);
        out.row(i) = out_Xi;
    }
}

void recall_nonlinear(mat X, mat W1, mat W2, vec b1, vec b2, mat &out){
    for(int i=0; i<size(X)[0]; i++) {
        vec out_h = zeros <vec> (size(W1)[0]);
        vec net_h = trans(X.row(i) * W1) + b1;
        sigmoid(out_h, net_h);

        vec out_o = zeros <vec> (size(W2)[0]);
        vec net_o = trans(W2) * out_h + b2;
        sigmoid(out_o, net_o);
        out.row(i) = out_o;
    }
}

void recall_multi(mat X, field <mat> W, field <vec> b, mat &out){
    field <vec> out_Xi(size(W)[0]+1);
    for(int i=0; i<size(X)[0]; i++) {
        out_Xi(0,0) = trans(X.row(i));
        for(int j=0; j<size(W)[0]; j++) {
            vec net = trans(W(j,0)) * out_Xi(j,0) + b(j,0);
            sigmoid(out_Xi(j+1,0), net);
        }
        out.row(i) = conv_to_number(out_Xi(size(W)[0]));
    }
}

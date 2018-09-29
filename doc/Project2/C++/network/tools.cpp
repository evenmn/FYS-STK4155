#include <iostream>
#include <armadillo>
#include <cmath>

using namespace arma;


void sigmoid(vec &out, vec net) {
    out = 1/(1 + exp(-net));
}


void sig_der(vec &out_der, vec out) {
    int Size = size(out_der)[0];
    vec one = ones <vec> (Size);
    out_der = out % (one - out);
}


void conv_from_number(double number, rowvec &out) {
    // Convert a double to array of doubles between 0 and 1
    int int_num = round(number - 0.5);
    out(0) = number - int_num;
    int i = 1;
    while (int_num != 0) {
        out(i) = (int_num % 10)/10.;
        int_num /= 10;
        i++;
    }
}

double conv_to_number(vec in) {
    // Convert an array of doubles between 0 and 1 to double
    double number;
    int i = 1;
    for(auto element : in) {
        number += element * i;
        i *= 10;
    }
    return number;
}

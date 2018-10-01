#pragma once
#include <armadillo>

void sigmoid(arma::vec &out, arma::vec net);
void sig_der(arma::vec &out_der, arma::vec out);
void conv_from_number(double number, arma::rowvec &out);
double conv_to_number(arma::vec in);

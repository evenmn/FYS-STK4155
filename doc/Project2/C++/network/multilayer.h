#pragma once
#include <armadillo>

void multi(arma::mat X, arma::mat t0, arma::field <arma::mat> &W, arma::field <arma::vec> &b, int h[], int H, int T, double eta);

#pragma once
#include <armadillo>

void linear(arma::mat X, arma::mat t, arma::mat &W, arma::vec &b, int T, double eta);
void nonlinear(arma::mat X, arma::mat t, arma::mat &W1, arma::mat &W2, arma::vec &b1, arma::vec &b2, int T, int H, double eta);

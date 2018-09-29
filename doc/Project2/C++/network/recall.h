#pragma once
#include <armadillo>

void recall_linear(arma::mat X, arma::mat W, arma::vec b, arma::mat &out);
void recall_nonlinear(arma::mat X, arma::mat W1, arma::mat W2, arma::vec b1, arma::vec b2, arma::mat &out);
void recall_multi(arma::mat X, arma::field <arma::mat> W, arma::field <arma::vec> b, arma::mat &out);

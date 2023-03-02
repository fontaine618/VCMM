#include "RcppArmadillo.h"
#include "VCMMSavedModel.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

VCMMSavedModel::VCMMSavedModel(){}

VCMMSavedModel::VCMMSavedModel(
  arma::mat a,
  arma::mat b,
  arma::rowvec t0,
  double alpha, double lambda, double ebic_factor, double kernel_scale,
  double objective, double apllk, double amllk, double bic, double ebic, 
  double rss, double parss, double predparss, double penalty, 
  double df_vc, double df_kernel, double sig2  
){
  this->a = a;
  this->b = b;
  this->t0 = t0;
  this->alpha = alpha;
  this->lambda = lambda;
  this->ebic_factor = ebic_factor;
  this->objective = objective;
  this->apllk = apllk;
  this->amllk = amllk;
  this->bic = bic;
  this->ebic = ebic;
  this->rss = rss;
  this->parss = parss;
  this->predparss = predparss;
  this->penalty = penalty;
  this->df_vc = df_vc;
  this->df_kernel = df_kernel;
  this->sig2 = sig2;
  this->kernel_scale = kernel_scale;
}

Rcpp::List VCMMSavedModel::to_RcppList(){
  return Rcpp::List::create(
    Rcpp::Named("a", this->a),
    Rcpp::Named("b", this->b),
    Rcpp::Named("t0", this->t0),
    Rcpp::Named("alpha", this->alpha),
    Rcpp::Named("lambda", this->lambda),
    Rcpp::Named("objective", this->objective),
    Rcpp::Named("apllk", this->apllk),
    Rcpp::Named("amllk", this->amllk),
    Rcpp::Named("bic", this->bic),
    Rcpp::Named("ebic", this->ebic),
    Rcpp::Named("ebic_factor", this->ebic_factor),
    Rcpp::Named("rss", this->rss),
    Rcpp::Named("parss", this->parss),
    Rcpp::Named("penalty", this->penalty),
    Rcpp::Named("df_vc", this->df_vc),
    Rcpp::Named("df_kernel", this->df_kernel),
    Rcpp::Named("sig2", this->sig2),
    Rcpp::Named("predparss", this->predparss),
    Rcpp::Named("kernel_scale", this->kernel_scale)
  );
}

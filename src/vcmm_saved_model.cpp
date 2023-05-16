#include "RcppArmadillo.h"
#include "VCMMSavedModel.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

VCMMSavedModel::VCMMSavedModel(){}

VCMMSavedModel::VCMMSavedModel(
  arma::mat a,
  arma::mat b,
  arma::rowvec t0,
  double alpha, double lambda, double kernel_scale,
  double objective, double mllk, double aic, double bic, 
  double rss, double parss, double predparss, double penalty, 
  double df_vc, double aic_kernel, double bic_kernel, double sig2, double re_ratio  
){
  this->a = a;
  this->b = b;
  this->t0 = t0;
  this->alpha = alpha;
  this->lambda = lambda;
  this->objective = objective;
  this->mllk = mllk;
  this->aic = aic;
  this->bic = bic;
  this->rss = rss;
  this->parss = parss;
  this->predparss = predparss;
  this->penalty = penalty;
  this->df_vc = df_vc;
  this->aic_kernel = aic_kernel;
  this->bic_kernel = bic_kernel;
  this->sig2 = sig2;
  this->re_ratio = re_ratio;
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
    Rcpp::Named("mllk", this->mllk),
    Rcpp::Named("aic", this->aic),
    Rcpp::Named("bic", this->bic),
    Rcpp::Named("rss", this->rss),
    Rcpp::Named("parss", this->parss),
    Rcpp::Named("penalty", this->penalty),
    Rcpp::Named("df_vc", this->df_vc),
    Rcpp::Named("aic_kernel", this->aic_kernel),
    Rcpp::Named("bic_kernel", this->bic_kernel),
    Rcpp::Named("sig2", this->sig2),
    Rcpp::Named("re_ratio", this->re_ratio),
    Rcpp::Named("predparss", this->predparss),
    Rcpp::Named("kernel_scale", this->kernel_scale)
  );
}

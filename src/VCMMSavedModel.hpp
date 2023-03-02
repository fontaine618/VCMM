#include "RcppArmadillo.h"
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef VCMMSavedModel_hpp
#define VCMMSavedModel_hpp

class VCMMSavedModel {
  
public:
  
  arma::mat a;
  arma::mat b;
  arma::rowvec t0;
  double alpha, lambda, ebic_factor, kernel_scale;
  double objective, apllk, amllk, bic, ebic, rss, parss, predparss, penalty, df_vc, df_kernel, sig2;
  
  VCMMSavedModel();  // need this to create empty lists
  VCMMSavedModel(
    arma::mat a,
    arma::mat b,
    arma::rowvec t0,
    double alpha, double lambda, double ebic_factor, double kernel_scale,
    double objective, double apllk, double amllk, double bic, double ebic, 
    double rss, double parss, double predparss, double penalty, 
    double df_vc, double df_kernel, double sig2  
  );
  
  Rcpp::List to_RcppList();
  
};

#endif
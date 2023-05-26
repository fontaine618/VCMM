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
  double objective, mllk, bic, aic, rss, parss, cv_score, penalty, df_vc, aic_kernel, bic_kernel, sig2, re_ratio;
  
  VCMMSavedModel();  // need this to create empty lists
  VCMMSavedModel(
    arma::mat a,
    arma::mat b,
    arma::rowvec t0,
    double alpha, double lambda, double kernel_scale,
    double objective, double mllk, double aic, double bic, 
    double rss, double parss, double cv_score, double penalty, 
    double df_vc, double aic_kernel, double bic_kernel, double sig2, double re_ratio  
  );
  
  Rcpp::List to_RcppList();
  
};

#endif
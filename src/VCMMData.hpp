#include "RcppArmadillo.h"
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef VCMMData_hpp
#define VCMMData_hpp

class VCMMData {
  
public:
  
  std::vector<arma::mat> p, w, x, u, z;
  std::vector<arma::colvec> y, t;
  int px, pu, q, nt;
  arma::rowvec t0;
  
  VCMMData(
    const arma::colvec & response,
    const arma::ucolvec & subject,
    const arma::colvec & response_time,
    const arma::mat & random_design,
    const arma::colvec & vcm_covariates,
    const arma::mat & fixed_covariates,
    const arma::rowvec & estimated_time,
    const double kernel_scale,
    const double mult
  );
  
};

#endif
#include "RcppArmadillo.h"
//[[Rcpp::depends(RcppArmadillo)]]

#ifndef VCMMData_hpp
#define VCMMData_hpp

class VCMMData {
  
public:
  
  std::vector<arma::mat> p, w, x, u, i;
  std::vector<arma::colvec> y, t;
  int px, pu, nt, n, N;
  arma::rowvec t0;
  double kernel_scale;
  arma::uvec foldid;
  
  VCMMData(
    const arma::colvec & response,
    const arma::ucolvec & subject,
    const arma::colvec & response_time,
    const arma::mat & vcm_covariates,
    const arma::mat & fixed_covariates,
    const arma::rowvec & estimated_time,
    const double kernel_scale,
    const bool random_effect
  );
  
  VCMMData(
    const std::vector<arma::colvec> & y,
    const std::vector<arma::colvec> & t,
    const std::vector<arma::mat> & x,
    const std::vector<arma::mat> & u,
    const std::vector<arma::mat> & p,
    const std::vector<arma::mat> & i,
    const std::vector<arma::mat> & w,
    const arma::rowvec & t0,
    const double kernel_scale
  );
  
  void update_weights(const double scale);
  
  void prepare_folds(uint nfolds);
  
  VCMMData get(arma::uvec ids);
  
  VCMMData get_fold(uint fold);
  
  VCMMData get_other_folds(uint fold);
  
  VCMMData resample();
  
};

#endif
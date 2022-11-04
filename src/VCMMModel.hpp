#include "RcppArmadillo.h"
#include <vector>
//[[Rcpp::depends(RcppArmadillo)]]
#include "VCMMData.hpp"
#include <math.h>

#ifndef VCMMModel_hpp
#define VCMMModel_hpp

class VCMMModel {
  
public:
  arma::mat b;
  arma::mat a;
  double alpha, lambda;
  int px, pu, q, nt;
  double La, Lb, momentum;
  
  VCMMModel(
    const int px,
    const int pu,
    const int nt,
    const int q,
    const double alpha,
    const double lambda
  );
  
  std::vector<arma::mat> linear_predictor(
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U
  );

  std::vector<arma::mat> residuals(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U
  );

  double loss(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  double penalty();
  
  std::vector<arma::mat> gradients(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  std::vector<std::vector<arma::mat>> _hessian_blocks (
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  arma::mat _hessian_from_blocks(
      arma::mat hessian_a,
      std::vector<arma::mat> hessian_b,
      std::vector<arma::mat> hessian_ab
  );
  
  arma::mat hessian (
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  arma::rowvec proximal(
    const arma::rowvec & b
  );
  
  void step(
      const std::vector<arma::mat> &gradients
  );
  
  void compute_lipschitz_constants(
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  double compute_lambda_max(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P,
      uint max_iter,
      double rel_tol
  );
  
  double fit(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P,
      uint max_iter,
      double rel_tol
  );
  
};

#endif


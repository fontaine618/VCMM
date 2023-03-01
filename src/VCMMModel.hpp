#include "RcppArmadillo.h"
#include <vector>
//[[Rcpp::depends(RcppArmadillo)]]
#include "VCMMData.hpp"
#include <math.h>

#ifndef VCMMModel_hpp
#define VCMMModel_hpp

class VCMMModel {
  
public:
  arma::mat b, tmpb;
  arma::mat a, tmpa;
  double alpha, lambda;
  uint px, pu, q, nt, max_iter;
  double La, Lb, momentum, rel_tol, cLa, cLb;
  arma::rowvec t0;
  double objective, rss, parss, pllk, apllk, df_kernel, amllk, bic, ebic, predparss;
  double sig2, sig2marginal, sig2profile;
  arma::mat Sigma;
  double ebic_factor;
  
  VCMMModel(
    const uint px,
    const uint pu,
    const uint nt,
    const uint q,
    const double alpha,
    const double lambda,
    const arma::rowvec &t0,
    const double ebic_factor,
    const double rel_tol,
    const uint max_iter
  );
  
  std::vector<arma::mat> linear_predictor(
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U
  );
  
  std::vector<arma::colvec> linear_predictor_at_observed_time(
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I
  );
  
  std::vector<arma::mat> residuals(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U
  );
  
  std::vector<arma::colvec> residuals_at_observed_time(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I
  );
  
  std::vector<arma::colvec> precision_adjusted_residuals(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  double loss(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  double profile_loglikelihood(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  double approximate_profile_loglikelihood(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  double approximate_marginal_loglikelihood(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  double compute_rss(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  double compute_parss(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  double penalty();
  
  uint df_vc();
  
  double compute_df_kernel(const std::vector<arma::mat> & W);
  
  std::vector<arma::mat> gradients(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  void backtracking_accelerated_proximal_gradient_step(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P,
      const double previous_obj
  );
  
  void accelerated_proximal_gradient_step(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  void monotone_accelerated_proximal_gradient_step(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  void proximal_gradient_step(
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
  
  arma::rowvec proximal_L1(
      const arma::rowvec & b,
      const double m
  );

  arma::rowvec proximal_L2(
      const arma::rowvec & s,
      const double m
  );

  std::vector<arma::mat> precision(
      const std::vector<arma::mat> & Z
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
      const std::vector<arma::mat> & I,
      uint max_iter
  );
  
  void fit(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P,
      const std::vector<arma::mat> & I,
      uint max_iter
  );
  

  void estimate_parameters(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & P,
      const std::vector<arma::mat> & Z,
      const std::vector<arma::mat> & I,
      const uint max_iter
  );
  
  void compute_statistics(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & Z,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P,
      const double kernel_scale
  );
  
  void compute_test_statistics(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & I,
      const std::vector<arma::mat> & P
  );
  
  void compute_ics(
    const uint n,
    const double h
  );
  
  Rcpp::List save();
  
  std::vector<Rcpp::List> grid_search(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      const double lambda_factor,
      uint n_lambda,
      VCMMData test
  );
  
  std::vector<Rcpp::List> grid_search(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      const double lambda_factor,
      uint n_lambda
  );
  
  std::vector<Rcpp::List> path(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      arma::uvec restart,
      VCMMData test
  );
  
  std::vector<Rcpp::List> path(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      arma::uvec restart
  );
  
  std::vector<Rcpp::List> orthogonal_search(
      VCMMData data,
      arma::vec kernel_scale,
      const double kernel_scale_factor,
      uint n_kernel_scale,
      arma::vec lambda,
      const double lambda_factor,
      uint n_lambda,
      uint max_tounds
  );
  
};

#endif


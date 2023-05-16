#include "RcppArmadillo.h"
#include <vector>
//[[Rcpp::depends(RcppArmadillo)]]
#include "VCMMData.hpp"
#include "VCMMSavedModel.hpp"
#include <math.h>



#ifndef VCMMModel_hpp
#define VCMMModel_hpp

class VCMMModel {
  
public:
  arma::mat b, tmpb;
  arma::mat a, tmpa;
  double alpha, lambda, kernel_scale;
  uint px, pu, nt, max_iter;
  double La, Lb, momentum, rel_tol, cLa, cLb;
  arma::rowvec t0;
  double objective, rss, parss, aic_kernel, bic_kernel, mllk, aic, bic, ebic, predparss;
  double sig2, re_ratio;
  arma::mat lasso_weights;
  arma::colvec grplasso_weights;
  bool penalize_intercept, progress_bar, estimate_variance_components, random_effect;
  
  VCMMModel(
    const uint px,
    const uint pu,
    const uint nt,
    const bool random_effect,
    const double alpha,
    const double lambda,
    const arma::rowvec &t0,
    const double rel_tol,
    const uint max_iter,
    const bool penalize_intercept,
    const bool estimate_variance_components,
    const bool progress_bar
  );
  
  VCMMSavedModel save();
  
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
  
  // double profile_loglikelihood(
  //     const std::vector<arma::colvec> & Y,
  //     const std::vector<arma::mat> & X,
  //     const std::vector<arma::mat> & U,
  //     const std::vector<arma::mat> & I,
  //     const std::vector<arma::mat> & P
  // );
  
  // double approximate_profile_loglikelihood(
  //     const std::vector<arma::colvec> & Y,
  //     const std::vector<arma::mat> & X,
  //     const std::vector<arma::mat> & U,
  //     const std::vector<arma::mat> & I,
  //     const std::vector<arma::mat> & P
  // );
  
  double marginal_loglikelihood(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
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
  
  double localized_parss(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  double logdet_global(
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  std::vector<arma::mat> precision_global(
      const std::vector<arma::mat> & W,
      const std::vector<arma::mat> & P
  );
  
  std::vector<arma::mat> total_weight(const std::vector<arma::mat> & W);
  
  double penalty();
  
  uint df_vc();
  
  arma::rowvec active();
  arma::rowvec effective_sample_size(const std::vector<arma::mat> & W, double kernel_scale);
  void compute_df_kernel(const std::vector<arma::mat> & W, double kernel_scale);
  
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
  
  arma::mat proximal_asgl(
      const arma::mat & b
  );
  
  arma::rowvec proximal_L1L2_row(
      const arma::rowvec & b,
      const arma::rowvec & m1,
      const double m2
  );
  
  arma::mat proximal_L1L2(
      const arma::mat & b,
      const arma::mat & m1,
      const arma::colvec m2
  );
  
  arma::rowvec proximal_L1(
      const arma::rowvec & b,
      const arma::rowvec & m
  );

  arma::rowvec proximal_L2(
      const arma::rowvec & s,
      const double m
  );

  void update_precision(std::vector<arma::mat> & P);
  
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
      std::vector<arma::mat> & P,
      uint max_iter
  );
  
  void fit(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      std::vector<arma::mat> & P,
      uint max_iter
  );
  

  void update_parameters(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
      const std::vector<arma::mat> & W,
      std::vector<arma::mat> & P
  );
  
  void compute_statistics(
      const std::vector<arma::colvec> & Y,
      const std::vector<arma::mat> & X,
      const std::vector<arma::mat> & U,
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
  
  void compute_ics();
  
  std::vector<VCMMSavedModel> grid_search(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      const double lambda_factor,
      uint n_lambda,
      VCMMData test,
      double adaptive
  );
  
  std::vector<VCMMSavedModel> grid_search(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      const double lambda_factor,
      uint n_lambda,
      double adaptive
  );
  
  std::vector<VCMMSavedModel> path(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      arma::uvec restart,
      VCMMData test,
      double adaptive
  );
  
  std::vector<VCMMSavedModel> path(
      VCMMData data,
      arma::vec kernel_scale,
      arma::vec lambda,
      arma::uvec restart,
      double adaptive
  );
  
  void compute_penalty_weights(
      VCMMData data,
      const double adaptive 
  );
  
  void unpenalize_intercept();
  
  std::vector<VCMMSavedModel> orthogonal_search(
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



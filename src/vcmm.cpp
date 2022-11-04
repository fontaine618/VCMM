#include "RcppArmadillo.h"
#include "VCMMData.hpp"
#include "VCMMModel.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export()]]
Rcpp::List VCMM(
    const arma::colvec & response,
    const arma::ucolvec & subject,
    const arma::colvec & response_time,
    const arma::mat & random_design,
    const arma::colvec & vcm_covariates,
    const arma::mat & fixed_covariates,
    const arma::rowvec & estimated_time,
    const double kernel_scale,
    const double alpha,
    const double lambda,
    const uint max_iter,
    const double mult
){
  VCMMData data = VCMMData(
    response, 
    subject,
    response_time,
    random_design,
    vcm_covariates,
    fixed_covariates,
    estimated_time,
    kernel_scale,
    mult
  );
  VCMMModel model = VCMMModel(
    2, 
    fixed_covariates.n_cols,
    estimated_time.n_elem,
    random_design.n_cols,
    alpha,
    lambda
  );
  Rcpp::Rcout << "Initialized data and models\n";
  
  model.compute_lipschitz_constants(data.x, data.u, data.w, data.p);
  
  double lambda_max = model.compute_lambda_max(data.y, data.x, data.u, data.w, data.p, max_iter, 1e-5);
  Rcpp::Rcout << "lambda max: " << lambda_max << "\n";
  
  model.lambda = lambda;
  double loss = model.fit(data.y, data.x, data.u, data.w, data.p, max_iter, 1e-5);
  
  return Rcpp::List::create(
    Rcpp::Named("a", model.a),
    Rcpp::Named("b", model.b),
    Rcpp::Named("X", data.x),
    Rcpp::Named("W", data.w),
    Rcpp::Named("P[0]", data.p[0])
  );
}
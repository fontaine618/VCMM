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
    double kernel_scale,
    const double alpha,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda,
    const uint max_iter,
    const double mult
){
  Rcpp::Rcout << "[VCMM] Initializing data and models ...";
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
    1e6,
    estimated_time
  );
  Rcpp::Rcout << "done.\n";
  
  Rcpp::Rcout << "[VCMM] Computing Lipschitz constants ...";
  model.compute_lipschitz_constants(data.x, data.u, data.w, data.p);
  Rcpp::Rcout << "done.\n";
  
  if(lambda.n_elem == 0){
    Rcpp::Rcout << "[VCMM] Computing maximum regularization parameter \n";
    double lambda_max = model.compute_lambda_max(data.y, data.x, data.u, data.w, data.p, data.i, max_iter, 1e-5);
    lambda = arma::logspace(log10(lambda_max*lambda_factor), log10(lambda_max), n_lambda);
    Rcpp::Rcout << "       done. (lambda max: " << lambda_max << ")\n";
  }
  lambda = arma::sort(lambda, "descend");
  n_lambda = lambda.n_elem;
  std::vector<Rcpp::List> models(n_lambda);
  
  for(uint l=0; l<n_lambda; l++){
    Rcpp::Rcout << "[VCMM] Lambda iteration " << l << " (lambda=" << lambda[l] << ")\n";
    model.lambda = lambda[l];
    model.fit(data.y, data.x, data.u, data.w, data.p, data.i, max_iter, 1e-5);
    
    Rcpp::Rcout << "       Estimating parameters ... \n";
    model.estimate_parameters(data.y, data.x, data.u, data.p, data.z, data.i, max_iter, 1e-5);
    Rcpp::Rcout << "       done.\n";
    
    Rcpp::Rcout << "       Computing final statistics ... ";
    model.compute_statistics(data.y, data.x, data.u, data.z, data.i, data.w, data.p);
    Rcpp::Rcout << "done.\n";
    
    models[l] = model.save();
  }
  
  return Rcpp::List::create(
    Rcpp::Named("models", models)
  );
}
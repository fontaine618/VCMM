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
    const uint n_kernel_scale, 
    const double alpha,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda,
    const uint max_iter,
    const double mult
){
  Rcpp::Rcout << "Initialized data and models ...";
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
  
  Rcpp::Rcout << "Computing Lipschitz constants ...\n";
  model.compute_lipschitz_constants(data.x, data.u, data.w, data.p);
  Rcpp::Rcout << "done.\n";
  
  if(n_kernel_scale > 0){
    Rcpp::Rcout << "Computing optimal kernel scale...\n";
    double h_max = estimated_time.max() - estimated_time.min();
    double h_min = exp(1.5) / data.n;
    arma::vec h_sequence = arma::logspace(log10(h_max), log10(h_min), n_kernel_scale);
    model.lambda = 0.;
    double best_bic = 1e10;
    double best_h;
    for(uint i=0; i<h_sequence.n_elem; i++){
      double h = h_sequence[i];
      data.update_weights(h);
      model.compute_lipschitz_constants(data.x, data.u, data.w, data.p);
      model.fit(data.y, data.x, data.u, data.w, data.p, max_iter, 1e-5);
      double bic = model.rss + model.active() * log(data.n * h) / (data.n * h);
      if(bic < best_bic){
        best_bic = bic;
        best_h = h;
      }
      Rcpp::Rcout << "    h=" << h << " BIC=" << bic << "\n";
    }
    Rcpp::Rcout << "done. (best scale=" << best_h << ")\n";
    data.update_weights(best_h);
  }
  
  if(lambda.n_elem == 0){
    Rcpp::Rcout << "Computing maximum regularization parameter ...\n";
    double lambda_max = model.compute_lambda_max(data.y, data.x, data.u, data.w, data.p, max_iter, 1e-5);
    lambda = arma::logspace(log10(lambda_max*lambda_factor), log10(lambda_max), n_lambda);
    Rcpp::Rcout << "done. (lambda max: " << lambda_max << ")\n";
  }
  lambda = arma::sort(lambda, "descend");
  n_lambda = lambda.n_elem;
  std::vector<Rcpp::List> models(n_lambda);
  
  for(uint l=0; l<n_lambda; l++){
    Rcpp::Rcout << "Lambda iteration " << l << " (lambda=" << lambda[l] << ")\n";
    model.lambda = lambda[l];
    model.fit(data.y, data.x, data.u, data.w, data.p, max_iter, 1e-5);
    // model.update_parameters(data.y, data.x, data.u, data.z, data.w, data.p);
    for(uint i=0; i<10; i++) {
      model.update_parameters(data.y, data.x, data.u, data.z, data.w);
    }
    models[l] = model.save();
  }
  
  return Rcpp::List::create(
    Rcpp::Named("models", models)
  );
}
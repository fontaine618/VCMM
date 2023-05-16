#include "RcppArmadillo.h"
#include "VCMMData.hpp"
#include "VCMMModel.hpp"
#include "VCMMSavedModel.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::export()]]
Rcpp::List VCMMBoot(
    const arma::colvec & response,
    const arma::ucolvec & subject,
    const arma::colvec & response_time,
    const arma::mat & vcm_covariates,
    const arma::mat & fixed_covariates,
    const arma::rowvec & estimated_time,
    bool random_effect,
    bool estimate_variance_components,
    const double kernel_scale,
    const double alpha,
    const double lambda,
    const float adaptive,
    const bool penalize_intercept,
    const uint max_iter,
    const double rel_tol,
    const int n_samples,
    const bool progress_bar
){
  Rcpp::Rcout << "[VCMMBoot] Initializing data and models ...";
  VCMMData data = VCMMData(
    response, 
    subject,
    response_time,
    vcm_covariates,
    fixed_covariates,
    estimated_time,
    kernel_scale,
    random_effect
  );
  
  VCMMModel model = VCMMModel(
    vcm_covariates.n_cols, 
    fixed_covariates.n_cols,
    estimated_time.n_elem,
    random_effect,
    alpha,
    lambda,
    estimated_time,
    rel_tol,
    max_iter,
    penalize_intercept,
    estimate_variance_components,
    progress_bar
  );
  
  Rcpp::Rcout << "done.\n";
  
  std::vector<VCMMSavedModel> models(n_samples);
  
  Rcpp::Rcout << "[VCMMBoot] Starting bootstrap loop ...\n";
  Progress pbar(n_samples);
  for(uint b=0; b<n_samples; b++){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(b);
    
    VCMMData rdata = data.resample();
    
    model.compute_lipschitz_constants(rdata.x, rdata.u, rdata.w, rdata.p);
    if(adaptive > 0.) model.compute_penalty_weights(rdata, adaptive);
    model.lambda = lambda;  // the above two will change lambda
    
    model.fit(rdata.y, rdata.x, rdata.u, rdata.w, rdata.p, rdata.i, max_iter);
    VCMMSavedModel submodel = model.save();
    submodel.kernel_scale = data.kernel_scale;
    models[b] = submodel;
    
    pbar.increment();
  }
  Rcpp::Rcout << "done.\n";
  
  // get estimate with full data
  model.compute_lipschitz_constants(data.x, data.u, data.w, data.p);
  if(adaptive > 0.) model.compute_penalty_weights(data, adaptive);
  model.lambda = lambda;  // the above two will change lambda
  
  model.fit(data.y, data.x, data.u, data.w, data.p, data.i, max_iter);
  VCMMSavedModel submodel = model.save();
  submodel.kernel_scale = data.kernel_scale;
  
  std::vector<Rcpp::List> models_list(models.size());
  for(uint m=0; m<models.size(); m++) models_list[m] = models[m].to_RcppList();
  return Rcpp::List::create(
    Rcpp::Named("boot", models_list),
    Rcpp::Named("model", submodel.to_RcppList())
  );
}
#include "RcppArmadillo.h"
#include "VCMMData.hpp"
#include "VCMMModel.hpp"
#include "VCMMSavedModel.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>
#include <map>

std::map<std::string, int> tuning_strategy_to_int{
  {"grid_search", 0},
  {"orthogonal_search", 1},
  {"bisection", 2}
};
  
// [[Rcpp::export()]]
Rcpp::List VCMM(
    const arma::colvec & response,
    const arma::ucolvec & subject,
    const arma::colvec & response_time,
    const arma::mat & random_design,
    const arma::mat & vcm_covariates,
    const arma::mat & fixed_covariates,
    const arma::rowvec & estimated_time,
    const std::string tuning_strategy,
    arma::vec kernel_scale,
    const double kernel_scale_factor,
    uint n_kernel_scale,
    const double alpha,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda,
    const uint max_iter,
    const double mult,
    const double ebic_factor,
    const double rel_tol,
    const uint orthogonal_search_max_rounds,
    const uint bissection_max_evals,
    const uint nfolds 
){
  Rcpp::Rcout << "[VCMM] Initializing data and models ...";
  double h = pow(response.n_elem, -0.2);
  VCMMData data = VCMMData(
    response, 
    subject,
    response_time,
    random_design,
    vcm_covariates,
    fixed_covariates,
    estimated_time,
    h,
    mult
  );
  
  VCMMModel model = VCMMModel(
    2, 
    fixed_covariates.n_cols,
    estimated_time.n_elem,
    random_design.n_cols,
    alpha,
    0.,
    estimated_time,
    ebic_factor,
    rel_tol,
    max_iter
  );
  Rcpp::Rcout << "done.\n";
  
  std::vector<VCMMSavedModel> models;
  switch(tuning_strategy_to_int[tuning_strategy]){
  // Orthogonal search
  case 1: 
    // implicitly initialized at h
    // models = model.orthogonal_search(
    //   data,
    //   kernel_scale,
    //   kernel_scale_factor,
    //   n_kernel_scale,
    //   lambda,
    //   lambda_factor,
    //   n_lambda,
    //   orthogonal_search_max_rounds
    // );
    break;
  // Bisection
  case 2:
    break;
  // Grid Search (0)
  default: 
    // we first compute the range of kernel scale: this depends on h, so we do it outside
    if(kernel_scale.n_elem == 0){
      kernel_scale = arma::logspace(log10(h*kernel_scale_factor), log10(h/kernel_scale_factor), n_kernel_scale);
    }
    n_kernel_scale = kernel_scale.n_elem;
    kernel_scale = arma::sort(kernel_scale, "descend");
    
    models = model.grid_search(
      data, 
      kernel_scale,
      lambda,
      lambda_factor,
      n_lambda
    );
    break;
  }
  
  // CV: maybe move this inside switch
  if(nfolds > 0){
    data.prepare_folds(nfolds);
    // need to find all the lambdas
    arma::vec fitted_lambdas(models.size());
    for(uint m=0; m<models.size(); m++) fitted_lambdas[m] = models[m].lambda;
    arma::vec fitted_hs(models.size());
    for(uint m=0; m<models.size(); m++) fitted_hs[m] = models[m].kernel_scale;
    // Sequencing is
    // [h0, ..., h0, h1, ..., h1, ..., hn, ..., hn]
    // [l00, ..., l0n, l10, ..., l1n, ..., lm0, ..., lmn]
    // Need to compute indicator for restarts when new h to 
    arma::uvec restart(models.size(), arma::fill::zeros);
    restart[0] = 1;
    for(uint m=0; m<models.size()-1; m++){
      if(fitted_hs[m]!=fitted_hs[m+1]) restart[m+1] = 1;
    }
    // arma::join_rows(fitted_lambdas, fitted_hs, restart).print();
    
    arma::vec predparss(models.size(), arma::fill::zeros);
    for(uint fold = 0; fold<nfolds; fold++){
      Rcpp::Rcout << "[VCMM] CV fold " << fold+1 << "/" << nfolds << "\n";
      VCMMData train = data.get_other_folds(fold);
      VCMMData test = data.get_fold(fold);
      std::vector<VCMMSavedModel> cvmodels = model.path(train, fitted_hs, fitted_lambdas, restart, test);
      for(uint m=0; m<cvmodels.size(); m++) predparss(m) += (double)cvmodels[m].predparss;
    }
    for(uint m=0; m<predparss.n_elem; m++) models[m].predparss = predparss[m];
  }
  
  std::vector<Rcpp::List> models_list(models.size());
  for(uint m=0; m<models.size(); m++) models_list[m] = models[m].to_RcppList();
  return Rcpp::List::create(
    Rcpp::Named("models", models_list)
  );
}
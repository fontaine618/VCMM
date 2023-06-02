#include "RcppArmadillo.h"
#include "VCMMData.hpp"
#include "VCMMModel.hpp"
#include "VCMMSavedModel.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
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
    const arma::mat & vcm_covariates,
    const arma::mat & fixed_covariates,
    const arma::rowvec & estimated_time,
    bool random_effect,
    bool estimate_variance_components,
    double re_ratio,
    const std::string tuning_strategy,
    arma::vec kernel_scale,
    const double kernel_scale_factor,
    uint n_kernel_scale,
    const double alpha,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda,
    const float adaptive,
    const bool penalize_intercept,
    const uint max_iter,
    const double rel_tol,
    const uint nfolds,
    const int cv_seed,
    bool progress_bar
){
  Rcpp::Rcout << "[VCMM] Initializing data and models ...";
  double h = pow(response.n_elem, -0.2);
  VCMMData data = VCMMData(
    response, 
    subject,
    response_time,
    vcm_covariates,
    fixed_covariates,
    estimated_time,
    h,
    random_effect,
    re_ratio
  );
  
  VCMMModel model = VCMMModel(
    vcm_covariates.n_cols, 
    fixed_covariates.n_cols,
    estimated_time.n_elem,
    random_effect,
    alpha,
    0.,
    estimated_time,
    rel_tol,
    max_iter,
    penalize_intercept,
    estimate_variance_components,
    progress_bar
  );
  double re_ratio2 = (re_ratio < 0.) ? log(subject.n_elem): re_ratio;
  model.re_ratio = random_effect ? re_ratio2 : 0.;
  Rcpp::Rcout << "done.\n";
  
  
  
  // Adaptive SGL
  if(adaptive > 0.){
    Rcpp::Rcout << "[VCMM] Computing penalty weights for adaptive SGL ...";
    model.compute_lipschitz_constants(data.x, data.u, data.w, data.p);
    model.compute_penalty_weights(data, adaptive);
    Rcpp::Rcout << "done.\n";
  }
  
  
  std::vector<VCMMSavedModel> models;
  switch(tuning_strategy_to_int[tuning_strategy]){
  // Orthogonal search
  case 1: 
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
      n_lambda,
      adaptive
    );
    break;
  }
  
  // CV: maybe move this inside switch
  if(nfolds > 0){
    // arma::arma_rng::set_seed(cv_seed); "When called from R, the RNG seed has to be set at the R level via set.seed()"
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(cv_seed);
    
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
    
    arma::vec cv_score(models.size(), arma::fill::zeros);
    for(uint fold = 0; fold<nfolds; fold++){
      Rcpp::Rcout << "[VCMM] CV fold " << fold+1 << "/" << nfolds << "\n";
      VCMMData train = data.get_other_folds(fold);
      VCMMData test = data.get_fold(fold);
      std::vector<VCMMSavedModel> cvmodels = model.path(train, fitted_hs, fitted_lambdas, restart, test, adaptive);
      for(uint m=0; m<cvmodels.size(); m++) cv_score(m) += (double)cvmodels[m].cv_score;
    }
    for(uint m=0; m<cv_score.n_elem; m++) models[m].cv_score = cv_score[m];
  }
  
  std::vector<Rcpp::List> models_list(models.size());
  for(uint m=0; m<models.size(); m++) models_list[m] = models[m].to_RcppList();
  return Rcpp::List::create(
    Rcpp::Named("models", models_list)
  );
}
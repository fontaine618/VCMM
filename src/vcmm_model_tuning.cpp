#include "RcppArmadillo.h"
#include "VCMMModel.hpp"
#include "VCMMSavedModel.hpp"
#include "VCMMData.hpp"
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>


std::vector<VCMMSavedModel> VCMMModel::grid_search(
    VCMMData data,
    arma::vec kernel_scale,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda,
    VCMMData test,
    double adaptive
){
  arma::mat prev_a = this->a * 0.;
  arma::mat prev_b = this->b * 0.;
  
  if(lambda_factor<=0) n_lambda = lambda.n_elem;
  uint n_kernel_scale = kernel_scale.n_elem;
  std::vector<VCMMSavedModel> models(n_kernel_scale*n_lambda);
  uint n_models = 0;
  
  for(uint k=0; k<n_kernel_scale; k++){
    // warmstart to largest lambda from last iteration
    this->a = prev_a;
    this->b = prev_b;
    
    double h = kernel_scale[k];
    Rcpp::Rcout << "[VCMM] Setting kernel scale to " << h << " (Iteration " << k+1 << "/" << n_kernel_scale << ")...";
    data.update_weights(h);
    test.update_weights(h);
    Rcpp::Rcout << "done.\n";
    
    Rcpp::Rcout << "[VCMM] Computing Lipschitz constants ...";
    this->compute_lipschitz_constants(data.x, data.u, data.w, data.p);
    Rcpp::Rcout << "done. (La=" << this->La << ", Lb=" << this->Lb << ")\n";
    
    
    // Adaptive SGL
    if(adaptive > 0.){
      Rcpp::Rcout << "[VCMM] Computing penalty weights for adaptive SGL ...";
      this->compute_penalty_weights(data, adaptive);
      Rcpp::Rcout << "done.\n";
      this->a = prev_a;
      this->b = prev_b;
    }
    
    if(lambda_factor > 0){
      Rcpp::Rcout << "[VCMM] Computing maximum regularization parameter \n";
      double lambda_max = this->compute_lambda_max(data.y, data.x, data.u, data.w, data.p, data.i, max_iter);
      lambda = arma::logspace(log10(lambda_max*lambda_factor), log10(lambda_max), n_lambda);
      Rcpp::Rcout << "       done. (lambda max: " << lambda_max << ")\n";
    }
    lambda = arma::sort(lambda, "descend");
    n_lambda = lambda.n_elem;
    
    Rcpp::Rcout << "[VCMM] Starting lambda loop ...\n";
    Progress pbar(n_lambda);
    for(uint l=0; l<n_lambda; l++){
      // Rcpp::Rcout << "[VCMM] Lambda iteration " << l << " (lambda=" << lambda[l] << ")\n";
      this->lambda = lambda[l];
      this->fit(data.y, data.x, data.u, data.w, data.p, data.i, max_iter);
      this->estimate_parameters(data.y, data.x, data.u, data.p, data.i, max_iter);
      this->compute_statistics(data.y, data.x, data.u, data.i, data.w, data.p, data.kernel_scale);
      this->compute_test_statistics(test.y, test.x, test.u, test.i, test.p);
      VCMMSavedModel submodel = this->save();
      submodel.kernel_scale = data.kernel_scale;
      if(l==0){
        // store coefficients for next h loop
        prev_a = this->a;
        prev_b = this->b;
      }
      pbar.increment();
      models[n_models] = submodel;
      n_models++;
    }
    Rcpp::Rcout << "       ... done.\n";
  }
  
  return models;
}

std::vector<VCMMSavedModel> VCMMModel::grid_search(
    VCMMData data,
    arma::vec kernel_scale,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda,
    double adaptive
){
  return this->grid_search(
    data,
    kernel_scale,
    lambda, 
    lambda_factor, 
    n_lambda, 
    data,
    adaptive
  );
}




std::vector<VCMMSavedModel> VCMMModel::path(
    VCMMData data,
    arma::vec kernel_scale,
    arma::vec lambda,
    arma::uvec restart,
    VCMMData test,
    double adaptive
){
  this->a.zeros();
  this->b.zeros();
  arma::mat prev_a = this->a * 0.;
  arma::mat prev_b = this->b * 0.;
  uint n_models = kernel_scale.n_elem;
  std::vector<VCMMSavedModel> cvmodels(n_models);
  Rcpp::Rcout << "[VCMM] Starting path (" << n_models << " models) ... \n";
  Progress pbar(n_models);
  
  for(uint k=0; k<n_models; k++){
    if(restart[k]==1){ // this means we have a new kernel_scale
      data.update_weights(kernel_scale[k]);
      test.update_weights(kernel_scale[k]);
      // warmstart to largest lambda from last iteration
      this->a = prev_a;
      this->b = prev_b;
      // Lipschitz constant also depends on kernel scale
      this->compute_lipschitz_constants(data.x, data.u, data.w, data.p);
      // Adaptive weights needs to be updated
      if(adaptive > 0.) this->compute_penalty_weights(data, adaptive);
      this->a = prev_a;
      this->b = prev_b;
    }
    this->lambda = lambda[k];
    this->fit(data.y, data.x, data.u, data.w, data.p, data.i, max_iter);
    this->estimate_parameters(data.y, data.x, data.u, data.p, data.i, max_iter);
    this->compute_statistics(data.y, data.x, data.u, data.i, data.w, data.p, data.kernel_scale);
    this->compute_test_statistics(test.y, test.x, test.u, test.i, test.p);
    VCMMSavedModel submodel = this->save();
    submodel.kernel_scale = data.kernel_scale;
    pbar.increment();
    cvmodels[k] = submodel;
    if(k<n_models - 1){
      if(restart[k+1]==1){
        prev_a = this->a;
        prev_b = this->b;
      }
    }
  
  }
  Rcpp::Rcout << "       ... done.\n";
  
  return cvmodels;
}

std::vector<VCMMSavedModel> VCMMModel::path(
    VCMMData data,
    arma::vec kernel_scale,
    arma::vec lambda,
    arma::uvec restart,
    double adaptive
){
  return this->path(data, kernel_scale, lambda, restart, data, adaptive);
}
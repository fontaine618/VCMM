#include "RcppArmadillo.h"
#include "VCMMModel.hpp"
#include "VCMMData.hpp"
#include <math.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>


std::vector<Rcpp::List> VCMMModel::grid_search(
    VCMMData data,
    arma::vec kernel_scale,
    arma::vec lambda,
    const double lambda_factor,
    uint n_lambda
){
  arma::mat prev_a = this->a * 0.;
  arma::mat prev_b = this->b * 0.;
  
  if(lambda_factor<=0) n_lambda = lambda.n_elem;
  uint n_kernel_scale = kernel_scale.n_elem;
  std::vector<Rcpp::List> models(n_kernel_scale*n_lambda);
  uint n_models = 0;
  
  for(uint k=0; k<n_kernel_scale; k++){
    // warmstart to largest lambda from last iteration
    this->a = prev_a;
    this->b = prev_b;
    
    double h = kernel_scale[k];
    Rcpp::Rcout << "[VCMM] Setting kernel scale to " << h << " (Iteration " << k+1 << "/" << n_kernel_scale << ")...";
    data.update_weights(h);
    Rcpp::Rcout << "done.\n";
    
    Rcpp::Rcout << "[VCMM] Computing Lipschitz constants ...";
    this->compute_lipschitz_constants(data.x, data.u, data.w, data.p);
    Rcpp::Rcout << "done. (La=" << this->La << ", Lb=" << this->Lb << ")\n";
    
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
      this->estimate_parameters(data.y, data.x, data.u, data.p, data.z, data.i, max_iter);
      this->compute_statistics(data.y, data.x, data.u, data.z, data.i, data.w, data.p, data.kernel_scale);
      
      Rcpp::List submodel = this->save();
      submodel["kernel_scale"] = data.kernel_scale;
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


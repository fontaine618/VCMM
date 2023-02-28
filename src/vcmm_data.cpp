#include "RcppArmadillo.h"
#include "VCMMData.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

arma::mat rbf_kernel_weight_matrix(
    const arma::rowvec & t1,
    const arma::colvec & t2,
    const double scale
){
  arma::mat od(t2.n_rows, t1.n_cols); 
  
  od.each_col() = t2;
  od.each_row() -= t1;
  
  od = arma::exp(- arma::square(od / scale) ) / scale;
  
  return od;
}

std::vector<arma::mat> compute_precision_per_subject(
    const arma::uvec & subject,
    const arma::mat & random_design,
    double mult
){
  arma::uvec ids = unique(subject);
  std::vector<arma::mat> out(ids.n_elem);
  arma::ucolvec is;
  arma::mat zi;
  arma::mat zzt;
  arma::mat prec;
  if (mult < 0.) mult = log(subject.n_elem);
  
  for(uint i : ids){
    is = arma::find(subject==i);
    zi = random_design.rows(is);
    zzt = zi * zi.t();
    prec = arma::eye(arma::size(zzt)) + mult * zzt;
    out[i] = arma::inv(prec);
  }
  
  return out;
}

std::vector<arma::mat> to_list_by_subject(
    const arma::uvec & subject,
    const arma::mat & array
){
  arma::uvec ids = unique(subject);
  std::vector<arma::mat> out(ids.n_elem);
  arma::ucolvec is;
  arma::mat rows;
  for(uint i : ids){
    is = arma::find(subject==i);
    rows = array.rows(is);
    out[i] = rows;
  }
  return out;
}

std::vector<arma::colvec> to_list_by_subject(
    const arma::uvec & subject,
    const arma::colvec & array
){
  arma::uvec ids = unique(subject);
  std::vector<arma::colvec> out(ids.n_elem);
  arma::ucolvec is;
  arma::colvec rows;
  for(uint i : ids){
    is = arma::find(subject==i);
    rows = array.rows(is);
    out[i] = rows;
  }
  return out;
}

arma::mat compute_interpolation(
  const arma::colvec & response_time,
  const arma::rowvec & estimated_time
){
  arma::mat out(response_time.n_elem, estimated_time.n_elem);
  out.zeros();
  
  for(uint i=0; i<out.n_rows; i++){
    double t = response_time[i];
    // find where t fits in
    int which = -1;
    for(uint j=0; j<out.n_cols; j++){
      if(estimated_time[j] >= t){
        which = j;
        break;
      }
    }
    // the two edge cases: use first/last only
    if(which>=out.n_cols){
      out(i, which) = 1.;
      continue;
    }
    if(which<1){
      out(i, 0) = 1.;
      continue;
    }
    // middle case: linear interpolation
    double t0 = estimated_time[which-1];
    double t1 = estimated_time[which];
    out(i, which-1) = 1 - (t-t0) / (t1 - t0);
    out(i, which) = 1 - (t1-t) / (t1 - t0);
  }
  return out;
}

VCMMData::VCMMData(
  const arma::colvec & response, // n x 1 in R
  const arma::ucolvec & subject, // n x 1 in {0, 1, ..., N-1}
  const arma::colvec & response_time, // n x 1 in R, but normally in [0,1] is scale_time=T from outside
  const arma::mat & random_design, // n x q in R, but normally just a column of 1s for random intercept
  const arma::mat & vcm_covariates, // n x px in R, but typically 0/1 (vc intercept has to be added outside)
  const arma::mat & fixed_covariates, // n x pu in R, need to copy outside if constant, but allows changing covariates
  const arma::rowvec & estimated_time, // 1 x nt in R, though it is preferable if scaled to [0,1]
  const double kernel_scale,
  const double mult
){
  this->kernel_scale = kernel_scale;
  // // create design matrix for time-varying covariates (with intercept)
  // arma::mat covariates(response.n_rows, 2);
  // covariates.col(0).ones();
  // covariates.col(1) = vcm_covariates;
  
  // compute the weights
  arma::mat weights = rbf_kernel_weight_matrix(estimated_time, response_time, kernel_scale);
  this->w = to_list_by_subject(subject, weights);
  
  // compute the precision matrix of random effects
  // (using the approximation from Fan and Li, 2012)
  // NB mult<0 uses mult=log(N), N=nb. observations 
  this->p = compute_precision_per_subject(subject, random_design, mult);
  
  // to list format and initialize object
  this->y = to_list_by_subject(subject, response);
  this->t = to_list_by_subject(subject, response_time);
  this->z = to_list_by_subject(subject, random_design);
  this->u = to_list_by_subject(subject, fixed_covariates);
  this->x = to_list_by_subject(subject, vcm_covariates);
  this->i = to_list_by_subject(subject, compute_interpolation(response_time, estimated_time));
  this->t0 = estimated_time;
  
  // store dimensions for quick access
  this->px = vcm_covariates.n_cols;
  this->pu = fixed_covariates.n_cols;
  this->q = random_design.n_cols;
  this->nt = t0.n_elem;
  this->n = response.n_elem;
  
}

// To change the weights without reinitializing an object altogether
void VCMMData::update_weights(const double scale){
  this->kernel_scale = scale;
  for(uint i = 0; i<this->w.size(); i++){
    this->w[i] = rbf_kernel_weight_matrix(this->t0, this->t[i], scale);
  }
}
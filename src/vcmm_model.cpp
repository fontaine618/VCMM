#include "RcppArmadillo.h"
#include "VCMMModel.hpp"
#include "VCMMSavedModel.hpp"
#include <math.h>

// [[Rcpp::depends(RcppArmadillo)]]

VCMMModel::VCMMModel(
  const uint px,
  const uint pu,
  const uint nt,
  const uint q,
  const double alpha,
  const double lambda,
  const arma::rowvec &t0,
  const double ebic_factor,
  const double rel_tol,
  const uint max_iter
){
  this->px = px;
  this->pu = pu;
  this->nt = nt;
  this->q = q;
  this->alpha = alpha;
  this->lambda = lambda;
  this->t0 = t0;
  this->ebic_factor = ebic_factor;
  this->rel_tol = rel_tol;
  this->max_iter = max_iter;
  this->sig2 = 1.;
  this->Sigma = arma::eye(q, q);

  // initialize coefficients to 0
  arma::mat b(px, nt);
  b.zeros();
  arma::mat a(pu, 1);
  a.zeros();
  this->a = a;
  this->b = b;
  this->tmpb = b;  //storage for FISTA step
  this->tmpa = a;
  
  // initialize weights
  this->lasso_weights = arma::ones(px, nt);
  this->grplasso_weights = arma::ones(px) / sqrt(nt);
}

double VCMMModel::compute_lambda_max(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P,
    const std::vector<arma::mat> & I,
    uint max_iter
){
  // start by getting completely sparse solution to get gradients
  this->lambda = 1e6;
  this->fit(Y, X, U, W, P, I, max_iter);
  std::vector<arma::mat> gradients = this->gradients(Y, X, U, W, P);
  // find max lambda:
  // we start with the two bounds, for each of the two prox operators
  arma::mat gb = gradients[1];
  arma::mat b1 = -gb / this->Lb;
  double lambda_max = -1e10;
  double gnorm2, w1norm2, denum, num;
  for(uint j=0; j<this->px; j++){
    // L1 bound
    if(this->alpha > 0.){
      for(uint t=0;t<this->nt; t++){
        if(this->lasso_weights(j, t) > 0.){
          num = fabs(gb(j, t));
          denum = this->alpha * this->lasso_weights(j, t) + (1-this->alpha) * this->grplasso_weights[j];
          lambda_max = fmax(lambda_max, num / denum);
        }
      }
    }
    // L2 bound
    gnorm2 = arma::norm(gb.row(j), "fro");
    w1norm2 = arma::norm(this->lasso_weights.row(j), "fro");
    if(this->alpha < 1. && this->grplasso_weights[j] > 0.){
      lambda_max = fmax(lambda_max, gnorm2 / (this->alpha * w1norm2 + (1-this->alpha) * this->grplasso_weights[j]));
    }
  }
  // // this will overshoot a bit possibly, so we decrease until a prox update would be nonzero and backtrack one step
  // arma::mat m1 = this->lasso_weights * this->alpha * lambda_max / this->Lb;
  // arma::colvec m2 = this->grplasso_weights * (1 - this->alpha) * lambda_max / this->Lb;
  // b = this->proximal_L1L2(b1, m1, m2);
  // 
  // uint iter = 0;
  // double mult = 0.95; // we should end up within 5% of the tightest bound
  // while(arma::norm(b, "inf") < 1.e-10){
  //   iter++;
  //   if(iter>100) break;
  //   lambda_max *= mult;
  //   m1 = this->lasso_weights * this->alpha * lambda_max / this->Lb;
  //   m2 = this->grplasso_weights * (1 - this->alpha) * lambda_max / this->Lb;
  //   b = this->proximal_L1L2(b1, m1, m2);
  // }
  // if(iter > 0) lambda_max /= mult;// we went one iteration too far
  return lambda_max; 
}

void VCMMModel::fit(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P,
    const std::vector<arma::mat> & I,
    uint max_iter
){
  // std::vector<arma::mat> g;
  // reset variables for (B)FISTA
  this->tmpa = this->a;
  this->tmpb = this->b;
  this->cLa = this->La;
  this->cLb = this->Lb;
  this->momentum = 1.;
  double prev_obj = this->loss(Y, X, U, W, P) + this->penalty();
  for(uint iter=1; iter<=max_iter; iter++){
    
    this->proximal_gradient_step(Y, X, U, W, P);
    // this->accelerated_proximal_gradient_step(Y, X, U, W, P);
    // this->monotone_accelerated_proximal_gradient_step(Y, X, U, W, P);
    // this->backtracking_accelerated_proximal_gradient_step(Y, X, U, W, P, prev_obj);
    
    double obj = this->loss(Y, X, U, W, P) + this->penalty();
    double rel_change = (obj - prev_obj) / fabs(obj);
    // if(iter % 1 == 0) Rcpp::Rcout <<
    //   "       Iteration " << iter << " obj: " << obj << " rel_change: " << rel_change << "\n";
    prev_obj = obj;
    if(fabs(rel_change) < this->rel_tol && iter > 0) {
      // Rcpp::Rcout << "       Iteration " << iter << " obj: " << obj << " rel_change: " << rel_change << " (converged) \n";
      break;
    }
  }
  this->objective = prev_obj;
}

void VCMMModel::compute_statistics(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & Z,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P,
    const double kernel_scale
){
  this->rss = this->compute_rss(Y, X, U, I, P);
  this->parss = this->compute_parss(Y, X, U, I, P);
  this->apllk = this->approximate_profile_loglikelihood(Y, X, U, I, P);
  this->amllk = this->approximate_marginal_loglikelihood(Y, X, U, I, P);
  this->compute_df_kernel(W, kernel_scale);
  
  // uint n = 0;
  // for(uint i=0; i<Y.size(); i++) n += Y[i].n_elem;
  this->compute_ics();
}

void VCMMModel::compute_test_statistics(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  this->predparss = this->compute_parss(Y, X, U, I, P);
}


void VCMMModel::estimate_parameters(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & P,
    const std::vector<arma::mat> & Z,
    const std::vector<arma::mat> & I,
    uint max_iter
){
  double rss = 0.;
  double parss = 0.;
  uint n = 0;
  std::vector<arma::colvec> R = this->residuals_at_observed_time(Y, X, U, I);
  
  for(uint i=0; i<R.size(); i++){
    arma::colvec pri = P[i] * R[i];
    rss += arma::dot(R[i], pri);
    parss += arma::dot(pri, pri);
    n += Y[i].n_elem;
  }
  
  this->sig2marginal = rss / n;
  this->sig2profile = parss / n;
  this->sig2 = this->sig2marginal;
}

void VCMMModel::compute_penalty_weights(
    const VCMMData data,
    const double adaptive 
){
  this->lambda = 0.;
  this->compute_lipschitz_constants(data.x, data.u, data.w, data.p);
  this->fit(data.y, data.x, data.u, data.w, data.p, data.i, max_iter);
  this->lasso_weights = arma::pow(arma::abs(this->b), -adaptive);
  arma::colvec row_norms = arma::zeros(this->px);
  for(uint j=0; j<this->px; j++) row_norms[j] = arma::norm(this->b.row(j), 2);
  this->grplasso_weights = arma::pow(row_norms, -adaptive);
}

VCMMSavedModel VCMMModel::save(){
  return VCMMSavedModel(
    this->a,
    this->b,
    this->t0,
    this->alpha,
    this->lambda,
    this->ebic_factor,
    this->kernel_scale,
    this->objective,
    this->apllk,
    this->amllk,
    this->aic,
    this->bic,
    this->rss,
    this->parss,
    this->predparss,
    this->penalty(),
    this->df_vc(),
    this->aic_kernel,
    this->bic_kernel,
    this->sig2
  );
}


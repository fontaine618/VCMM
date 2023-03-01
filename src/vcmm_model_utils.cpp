#include "RcppArmadillo.h"
#include "VCMMModel.hpp"
#include <math.h>

// [[Rcpp::depends(RcppArmadillo)]]



std::vector<arma::mat> VCMMModel::linear_predictor(
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U
){
  std::vector<arma::mat> out(X.size());
  arma::mat eta;
  arma::colvec eta2;
  
  for(uint i=0; i<out.size(); i++){
    eta = X[i] * this->b;
    eta2 = U[i] * this->a;
    eta.each_col() += eta2;
    out[i] = eta;
  }
  
  return out;
}

std::vector<arma::colvec> VCMMModel::linear_predictor_at_observed_time(
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I
){
  std::vector<arma::colvec> out(X.size());
  arma::colvec eta;
  arma::colvec eta2;
  
  for(uint i=0; i<out.size(); i++){
    eta = arma::sum(X[i] % (I[i] * this->b.t()), 1);
    eta2 = U[i] * this->a;
    out[i] = eta + eta2;
  }
  
  return out;
}

std::vector<arma::mat> VCMMModel::residuals(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U
){
  std::vector<arma::mat> eta(Y.size()), R(Y.size());
  eta = this->linear_predictor(X, U);
  
  for(uint i=0; i<R.size(); i++){
    arma::mat etai = eta[i];
    etai.each_col() -= Y[i];
    R[i] = -etai;
  }
  
  return R;
}

std::vector<arma::colvec> VCMMModel::residuals_at_observed_time(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I
){
  std::vector<arma::colvec> eta(Y.size()), R(Y.size());
  eta = this->linear_predictor_at_observed_time(X, U, I);
  
  for(uint i=0; i<R.size(); i++){
    R[i] = Y[i] - eta[i];
  }
  
  return R;
}

std::vector<arma::colvec> VCMMModel::precision_adjusted_residuals(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  std::vector<arma::colvec> R(Y.size());
  R = this->residuals_at_observed_time(Y, X, U, I);
  
  for(uint i=0; i<R.size(); i++){
    R[i] = P[i] * R[i];
  }
  
  return R;
}

double VCMMModel::compute_parss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  double rss = 0.;
  std::vector<arma::colvec> R = this->residuals_at_observed_time(Y, X, U, I);
  
  for(uint i=0; i<R.size(); i++){
    rss += arma::dot(R[i], P[i] * R[i]);
  }
  
  return rss;
}

double VCMMModel::compute_rss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  double parss = 0.;
  std::vector<arma::colvec> R = this->precision_adjusted_residuals(Y, X, U, I, P);
  
  for(uint i=0; i<R.size(); i++){
    parss += arma::dot(R[i], R[i]);
  }
  
  return parss;
}

double VCMMModel::loss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> R(Y.size());
  R = this->residuals(Y, X, U);
  double loss = 0.;
  double sw = 0.;
  
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec rk = R[i].col(k);
      arma::colvec wk = W[i].col(k);
      loss += arma::dot(rk, (P[i] % (arma::sqrt(wk * wk.t()))) * rk);
      // sw += arma::accu(arma::sqrt(wk * wk.t()));
    }
    sw += Y[i].n_elem;
  }
  
  return 0.5 * loss / sw;
}


double VCMMModel::profile_loglikelihood(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  // This uses *any* precision matrix:
  // Use ->precision to get the true Ps beforehand
  // Note that we divide by sig2 in the RSS, so keep that in mind when passing P
  uint N = Y.size();
  uint n = 0;
  std::vector<arma::colvec> R(N);
  R = this->residuals_at_observed_time(Y, X, U, I);
  double pllk = 0.;
  double sig2 = this->sig2;
  
  // compute RSS / sigma^2
  for(uint i=0; i<N; i++){
    arma::colvec r = R[i];
    pllk += arma::dot(r, P[i] * r);
    n += r.size();
  }
  pllk /= sig2;
  
  // Add the other terms
  pllk += N * arma::log_det_sympd(2 * arma::datum::pi * this->Sigma);
  pllk += n * log(2 * arma::datum::pi * sig2);
  
  return -0.5 * pllk;
}


double VCMMModel::approximate_profile_loglikelihood(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  // This does not use this->Sigma, 
  uint N = Y.size();
  uint n = 0;
  double sig2 = 0.;
  double apllk = 0.;
  std::vector<arma::colvec> R(N);
  R = this->residuals_at_observed_time(Y, X, U, I);
  double parss = this->compute_parss(Y, X, U, I, P);
  for(uint i=0; i<Y.size(); i++){
    n += Y[i].n_elem;
    arma::colvec Pri = P[i] * R[i];
    sig2 += arma::dot(Pri, Pri);
  }
  sig2 /= n;
  apllk = n * log(2 * arma::datum::pi * sig2) + parss / sig2;
  return -0.5 * apllk;
}


double VCMMModel::approximate_marginal_loglikelihood(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & I,
    const std::vector<arma::mat> & P
){
  uint n = 0;
  double logdet = 0.;
  double sig2 = this->compute_parss(Y, X, U, I, P);
  for(uint i=0; i<Y.size(); i++){
    n += Y[i].n_elem;
    logdet += arma::log_det_sympd(P[i]);
  }
  sig2 /= n;
  
  return 0.5 * (logdet - n*log(sig2));
}



double VCMMModel::penalty(){
  double penalty;
  double l1, l2;
  
  l1 = arma::norm(this->b.row(1), 1);
  l2 = arma::norm(this->b.row(1), 2);
  penalty = this->alpha * l1 ;
  penalty += (1. - this->alpha) * sqrt(this->nt) * l2;
  
  return penalty * this->lambda;
}

std::vector<arma::mat> VCMMModel::gradients(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> R = this->residuals(Y, X, U);
  arma::mat grad_a(this->pu, 1);
  arma::mat grad_b(this->px, this->nt);
  grad_a.zeros();
  grad_b.zeros();
  double sw = 0.;
  
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec rk = R[i].col(k);
      arma::colvec wk = W[i].col(k);
      // sw += arma::accu(arma::sqrt(wk * wk.t()));
      arma::mat M = (P[i] % (arma::sqrt(wk * wk.t()))) * rk;
      grad_a += U[i].t() * M;
      grad_b.col(k) += X[i].t() * M;
    }
    sw += Y[i].n_elem;
  }
  return std::vector<arma::mat>{- grad_a / sw, - grad_b / sw};
}

std::vector<std::vector<arma::mat>> VCMMModel::_hessian_blocks(
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> hessian_b(this->nt);
  std::vector<arma::mat> hessian_ab(this->nt);
  arma::mat hessian_a(this->pu, this->pu);
  double sw = 0.;
  
  // initialize to 0
  hessian_a.zeros();
  for(uint i=0; i<this->nt; i++){
    hessian_b[i] = arma::zeros(this->px, this->px);
    hessian_ab[i] = arma::zeros(this->px, this->pu);
  }
  
  // compute hessian
  for(uint i=0; i<X.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = W[i].col(k);
      // sw += arma::accu(arma::sqrt(wk * wk.t()));
      arma::mat M = P[i] % (arma::sqrt(wk * wk.t()));
      hessian_a += U[i].t() * M * U[i];
      hessian_b[k] += X[i].t() * M * X[i];
      hessian_ab[k] += X[i].t() * M * U[i];
    }
    sw += X[i].n_rows;
  }
  
  // divide by total weight
  hessian_a /= sw;
  for(uint i=0; i<this->nt; i++){
    hessian_b[i] /= sw;
    hessian_ab[i] /= sw;
  }
  
  
  std::vector<std::vector<arma::mat>> out{
    {hessian_a},
    hessian_b,
    hessian_ab
  };
  return out;
}



arma::mat VCMMModel::_hessian_from_blocks(
    arma::mat hessian_a,
    std::vector<arma::mat> hessian_b,
    std::vector<arma::mat> hessian_ab
){
  uint dim = this->pu + this->nt * this->px;
  arma::mat hessian(dim, dim);
  hessian.zeros();
  
  if(this->pu > 0) hessian.submat(0, 0, this->pu - 1, this->pu - 1) = hessian_a;
  for(uint k=0; k<this->nt; k++){
    if(this->pu > 0){
      hessian.submat(0,
                     this->pu + k*this->px,
                     this->pu - 1,
                     this->pu + (k+1)*this->px - 1) = hessian_ab[k].t();
      hessian.submat(this->pu + k*this->px,
                     0,
                     this->pu + (k+1)*this->px - 1,
                     this->pu - 1) = hessian_ab[k];
    }
    hessian.submat(this->pu + k*this->px,
                   this->pu + k*this->px,
                   this->pu + (k+1)*this->px - 1,
                   this->pu + (k+1)*this->px - 1) = hessian_b[k];
  }
  return hessian;
}

arma::mat VCMMModel::hessian(
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<std::vector<arma::mat>> hessians = this->_hessian_blocks(X, U, W, P);
  
  return this->_hessian_from_blocks(
      hessians[0][0],
                 hessians[1],
                         hessians[2]
  );
}


void VCMMModel::compute_lipschitz_constants(
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<std::vector<arma::mat>> hessians = this->_hessian_blocks(X, U, W, P);
  arma::mat hessian = this->_hessian_from_blocks(
    hessians[0][0],
    hessians[1],
    hessians[2]
  );
  
  // Find heuristic values for La, Lb
  double La, Lb;
  arma::vec eigval;
  
  La = 0.;
  if(this->pu > 0){
    arma::eig_sym(eigval, hessians[0][0]);
    La = eigval.max();
  }
  
  Lb = 0;
  for(uint k=0; k<this->nt; k++){
    arma::eig_sym(eigval, hessians[1][k]);
    Lb = fmax(Lb, eigval.max());
  }
  
  arma::mat HmLI = hessian;
  for(uint j=0; j<this->pu; j++){
    HmLI(j, j) -= La;
  }
  for(uint t=0; t<this->nt; t++){
    for(uint j=0; j<this->px; j++){
      HmLI(this->pu + t*this->px + j, this->pu + t*this->px + j) -= Lb;
    }
  }
  arma::eig_sym(eigval, -HmLI);
  double minevalHmLI = eigval.min();
  
  uint iter=0;
  while(minevalHmLI<=0.){
    iter++;
    La *= 1.01;
    Lb *= 1.01;
    HmLI = hessian;
    for(uint j=0; j<this->pu; j++){
      HmLI(j, j) -= La;
    }
    for(uint t=0; t<this->nt; t++){
      for(uint j=0; j<this->px; j++){
        HmLI(this->pu + t*this->px + j, this->pu + t*this->px + j) -= Lb;
      }
    }
    arma::eig_sym(eigval, -HmLI);
    minevalHmLI = eigval.min();
    // Rcpp::Rcout << iter << " minevalHmLI: " << minevalHmLI << "\n";
  }
  // Rcpp::Rcout << " (" << iter << " iterations) ";
  // arma::eig_sym(eigval, hessian);
  // Rcpp::Rcout << " (min eval=" << eigval.min() << ") ";
  // // overwrite everything to just largest eigenvalue of the hessian:
  // arma::eig_sym(eigval, hessian);
  // double L = eigval.max();
  // L = 1000.;
  // La = L;
  // Lb = L;
  
  this->La = La;
  this->Lb = Lb;
  // Rcpp::Rcout << "La: " << this->La << " Lb: " << this->Lb << "\n";
}

arma::rowvec VCMMModel::proximal_L1(
    const arma::rowvec &b,
    const double m
){
  uint nt = b.n_elem;
  arma::rowvec s(nt);
  s = arma::abs(b) - m;
  s = arma::sign(b) % arma::clamp(s, 0., arma::datum::inf);
  return s;
}

arma::rowvec VCMMModel::proximal_L2(
    const arma::rowvec &s,
    const double m
){
  double sn = arma::norm(s);
  if(sn == 0.) sn = 1.; // to avoid dividing by 0, the value doesn't matter since s=0 in that case
  return fmax(1. - m/sn, 0.) * s;
}

arma::rowvec VCMMModel::proximal(
    const arma::rowvec &b
){
  double m1 = this->alpha * this->lambda / this->Lb;
  double m2 = sqrt(this->nt) * (1 - this->alpha) * this->lambda / this->Lb;
  return proximal_L2(proximal_L1(b, m1), m2);
}


std::vector<arma::mat> VCMMModel::precision(
    const std::vector<arma::mat> & Z
){
  uint N = Z.size();
  std::vector<arma::mat> P(N);
  for(uint i=0; i<N; i++){
    uint ni = Z[i].n_rows;
    arma::mat Pi = arma::eye(ni, ni);
    Pi += Z[i] * this->Sigma * Z[i].t() / this->sig2;
    P[i] = Pi.i();
  }
  return P;
}


void VCMMModel::compute_ics(const uint n, const double h){
  double nh = n * h;
  uint p = this->nt * this->px;
  uint df = this->df_vc();
  double bic_penalty = df * log(nh) / nh;
  double ebic_penalty = df * log(p) / nh;
  this->bic = -2*this->amllk + bic_penalty;
  this->ebic = this->bic + ebic_penalty;
}

uint VCMMModel::df_vc(){
  return arma::accu(this->b != 0.);
}

arma::rowvec VCMMModel::active(){
  arma::rowvec df(this->nt, arma::fill::zeros);
  for(uint s=0; s<this->nt; s++){
    df[s] = arma::accu(this->b.col(s) != 0.);
  }
  return df;
}

arma::rowvec VCMMModel::effective_sample_size(
    const std::vector<arma::mat> & W, 
    double kernel_scale
){
  // TODO: this assumes k(0) = 1
  arma::rowvec n(this->nt, arma::fill::zeros);
  for(uint i = 0; i < W.size(); i++){
    n += arma::sum(W[i], 0); // W[i] is ni x nt, dim=0 takes colsum
  }
  n *= kernel_scale; 
  return n;
}

double VCMMModel::compute_df_kernel(
    const std::vector<arma::mat> & W, 
    double kernel_scale
){
  arma::rowvec n = this->effective_sample_size(W, kernel_scale);
  arma::rowvec df = this->active(); 
  // for now, we assume 1/T as the contribution of each time point, 
  // we also assume k(0)=1
  double range = this->t0.max() - this->t0.min();
  double df_kernel = arma::dot(arma::log(n), df) * range / (this->nt * kernel_scale);
  return df_kernel;
}


void VCMMModel::backtracking_accelerated_proximal_gradient_step(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P,
    const double previous_obj
){
  double backtracking_factor = 1.1;
  std::vector<arma::mat> g = this->gradients(Y, X, U, W, P);
  double pt = this->momentum;
  double newt = 0.5 * (1. + sqrt(1 + 4.*pt*pt));
  double gap = 1.; // set to one to enter while loop below
  double La = this->La / backtracking_factor; // we divide because the loop will multiply the first one
  double Lb = this->Lb / backtracking_factor;
  arma::colvec propa;
  arma::mat propb;
  arma::colvec tmpa;
  arma::mat tmpb;
  arma::colvec preva = this->a;
  arma::mat prevb = this->b;
  double obj, quad;
  while(gap > 0.){
    La *= backtracking_factor;
    Lb *= backtracking_factor;
    // compute regular update using L
    tmpa = preva - g[0] / La;
    tmpb = prevb - g[1] / Lb;
    if(this->lambda > 0.) tmpb.row(1) = this->proximal(tmpb.row(1));
    // update stepsize and take momentum step
    propa = tmpa + (pt - 1.) * (tmpa - this->tmpa) / momentum;
    propb = tmpb + (pt - 1.) * (tmpb - this->tmpb) / momentum;
    // Compute obj and quadratic bound
    // Objective: evaluated at the proximal step (tmp)
    obj = this->loss(Y, X, U, W, P) + this->penalty();
    // Quadratic bound only depends on prox step (tmp) and previous point (prev)
    // First term: objective at previous point
    quad = previous_obj;
    // Second term: inner product between the two points and the gradient at the previous point
    quad += arma::dot(tmpa - preva, g[0]);
    quad += arma::accu((tmpb - prevb) % g[1]);
    // Third term: squared norm of difference
    quad += 0.5 * La * arma::accu(arma::pow(tmpa - preva, 2.));
    quad += 0.5 * Lb * arma::accu(arma::pow(tmpb - prevb, 2.));
    // Fourth term: penalty at prox step (tmp)
    this->a = tmpa;
    this->b = tmpb;
    quad += this->penalty();
    // want F < Q, so we look for gap=F-Q<0
    gap = obj - quad;
  }
  this->tmpa = tmpa;
  this->tmpb = tmpb;
  this->a = propa;
  this->b = propb;
  this->momentum = newt;
  this->cLa = La;
  this->cLb = Lb;
}

void VCMMModel::accelerated_proximal_gradient_step(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> g = this->gradients(Y, X, U, W, P);
  double pt = this->momentum;
  double newt = 0.5 * (1. + sqrt(1 + 4.*pt*pt));
  arma::colvec tmpa = this->a - g[0] / this->La;
  arma::mat tmpb = this->b - g[1] / this->Lb;
  if(this->lambda > 0.) tmpb.row(1) = this->proximal(tmpb.row(1));
  this->a = tmpa + (pt - 1.) * (tmpa - this->tmpa) / momentum;
  this->b = tmpb + (pt - 1.) * (tmpb - this->tmpb) / momentum;
  this->tmpa = tmpa;
  this->tmpb = tmpb;
  this->momentum = newt;
}

void VCMMModel::proximal_gradient_step(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> g = this->gradients(Y, X, U, W, P);
  arma::colvec tmpa = this->a - g[0] / this->La;
  arma::mat tmpb = this->b - g[1] / this->Lb;
  if(this->lambda > 0.) tmpb.row(1) = this->proximal(tmpb.row(1));
  this->a = tmpa;
  this->b = tmpb;
}

void VCMMModel::monotone_accelerated_proximal_gradient_step(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> g = this->gradients(Y, X, U, W, P);
  double pt = this->momentum;
  double newt = 0.5 * (1. + sqrt(1 + 4.*pt*pt));
  arma::colvec proposed_a = this->a - g[0] / this->La;
  arma::mat proposed_b = this->b - g[1] / this->Lb;
  if(this->lambda > 0.) tmpb.row(1) = this->proximal(tmpb.row(1));
  // compare objectives
  this->a = proposed_a;
  this->b = proposed_b;
  double proposed_obj = this->loss(Y, X, U, W, P) + this->penalty();
  this->a = this->tmpa;
  this->b = this->tmpb;
  double tmpobj = this->loss(Y, X, U, W, P) + this->penalty();
  arma::colvec tmpa = this->tmpa;
  arma::mat tmpb = this->tmpb;
  if(proposed_obj < tmpobj){
    // proposed is better
    tmpa = proposed_a;
    tmpb = proposed_b;
  } // otherwise tmp is better, so we do nothing
  this->a = tmpa + (pt - 1.) * (tmpa - this->tmpa) / momentum;
  this->b = tmpb + (pt - 1.) * (tmpb - this->tmpb) / momentum;
  this->tmpa = tmpa;
  this->tmpb = tmpb;
  this->momentum = newt;
}

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

std::vector<arma::colvec> VCMMModel::global_linear_predictor(
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  uint N = X.size();
  std::vector<arma::mat> eta(N), R(N);
  std::vector<arma::colvec> g_eta(N);
  eta = this->linear_predictor(X, U);
  std::vector<arma::mat> g_prec(N);
  
  for(uint i=0; i<N; i++){
    uint ni = P[i].n_rows;
    g_eta[i] = arma::zeros(ni);
    g_prec[i] = arma::zeros(ni, ni);
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = arma::sqrt(W[i].col(k));
      arma::mat Pk = (wk * wk.t()) % P[i];
      g_eta[i] += Pk * eta[i].col(k);
      g_prec[i] += Pk;
    }
    
    g_eta[i] = arma::solve(g_prec[i], g_eta[i]);
  }
  
  return g_eta;
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


std::vector<arma::colvec> VCMMModel::global_residuals(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::colvec> eta(Y.size()), R(Y.size());
  eta = this->global_linear_predictor(X, U, W, P);
  
  for(uint i=0; i<R.size(); i++){
    R[i] = Y[i] - eta[i];
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


std::vector<arma::mat> VCMMModel::precision_global(
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  // unscaled
  uint N = W.size();
  std::vector<arma::mat> WT = this->global_weight(W);
  std::vector<arma::mat> PG(N);
  for(uint i=0; i<N; i++){
    PG[i] = P[i] % WT[i];
  }
  return PG;
}

std::vector<arma::mat> VCMMModel::global_precision(
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  uint N = W.size();
  std::vector<arma::mat> g_prec(N);
  
  for(uint i=0; i<N; i++){
    uint ni = P[i].n_rows;
    g_prec[i] = arma::zeros(ni, ni);
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = arma::sqrt(W[i].col(k));
      g_prec[i] += (wk * wk.t()) % P[i];
    }
  }
  
  return g_prec;
}

std::vector<arma::mat> VCMMModel::global_weight(
    const std::vector<arma::mat> & W
){
  uint N = W.size();
  std::vector<arma::mat> GW(N);
  
  for(uint i=0; i<N; i++){
    uint ni = W[i].n_rows;
    GW[i] = arma::zeros(ni, ni);
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = arma::sqrt(W[i].col(k));
      GW[i] += wk * wk.t();
    }
  }
  
  return GW;
}




double VCMMModel::total_weight(
    const std::vector<arma::mat> & W
){
  uint N = W.size();
  double totweight = 0.;
  
  for(uint i=0; i<N; i++){
    uint ni = W[i].n_rows;
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = arma::sqrt(W[i].col(k));
      totweight += arma::accu(arma::pow(wk, 2));
    }
  }
  
  return totweight;
}

double VCMMModel::compute_rss(
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

double VCMMModel::localized_parss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  double parss = 0.;
  std::vector<arma::mat> R = this->residuals(Y, X, U);
  
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec rk = R[i].col(k) % arma::sqrt(W[i].col(k));
      parss += arma::dot(rk, P[i] * rk);
    }
  }
  return parss;
}

// double VCMMModel::compute_rss(
//     const std::vector<arma::colvec> & Y,
//     const std::vector<arma::mat> & X,
//     const std::vector<arma::mat> & U,
//     const std::vector<arma::mat> & I,
//     const std::vector<arma::mat> & P
// ){
//   double parss = 0.;
//   std::vector<arma::colvec> R = this->precision_adjusted_residuals(Y, X, U, I, P);
//   
//   for(uint i=0; i<R.size(); i++){
//     parss += arma::dot(R[i], R[i]);
//   }
//   
//   return parss;
// }

double VCMMModel::localized_rss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W
){
  double parss = 0.;
  std::vector<arma::mat> R = this->residuals(Y, X, U);
  
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec rk = R[i].col(k) % arma::sqrt(W[i].col(k));
      parss += arma::dot(rk, rk);
    }
  }
  return parss;
}

double VCMMModel::global_parss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  double quad = 0.;
  std::vector<arma::colvec> R = this->global_residuals(Y, X, U, W, P);
  std::vector<arma::mat> PG = this->precision_global(W, P);
  for(uint i=0; i<Y.size(); i++){
    quad += arma::dot(R[i], PG[i] * R[i]);
  }
  
  return quad;
}

double VCMMModel::loss(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> R = this->residuals(Y, X, U);
  double loss = 0.;
  double n = 0.;
  
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec rk = R[i].col(k) % arma::sqrt(W[i].col(k));
      loss += arma::dot(rk, P[i] * rk);
    }
    n += Y[i].n_elem;
  }
  
  return 0.5 * loss / n;
}

double VCMMModel::penalty(){
  double penalty = 0.;
  for(uint j=0; j<this->px; j++){
    penalty += this->alpha * arma::accu(arma::abs(this->b.row(j)) % this->lasso_weights.row(j));
    double l2j = arma::norm(this->b.row(j), 2);
    penalty += (1. - this->alpha) * this->grplasso_weights(j) * l2j;
  }
  return penalty * this->lambda;
}


uint VCMMModel::effective_sample_size(
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  uint n = 0;
  
  for(uint i=0; i<W.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = arma::sqrt(W[i].col(k));
      arma::mat PW = (wk * wk.t()) % P[i];
      arma::colvec evals = arma::eig_sym(PW);
      n += arma::accu(evals > 1e-6);
    }
  }
  return n;
}

double VCMMModel::localized_marginal_loglikelihood(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  double quad = 0.;
  double logdet = 0.;
  std::vector<arma::mat> R = this->residuals(Y, X, U);
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = arma::sqrt(W[i].col(k));
      arma::mat Pik = P[i] % (wk * wk.t());
      arma::colvec rk = R[i].col(k);
      quad += arma::dot(rk, Pik * rk);
      arma::colvec evals = arma::eig_sym(Pik / (this->sig2 * 2 * arma::datum::pi));
      evals = evals.elem(arma::find(evals > 1e-6));
      logdet += arma::accu(arma::log(evals));
    }
  }
  
  return 0.5 * (logdet - quad / this->sig2);
}

double VCMMModel::global_marginal_loglikelihood(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  double quad = 0.;
  double logdet = 0.;
  std::vector<arma::colvec> R = this->global_residuals(Y, X, U, W, P);
  std::vector<arma::mat> PG = this->precision_global(W, P);
  for(uint i=0; i<Y.size(); i++){
    quad += arma::dot(R[i], PG[i] * R[i]);
    logdet += arma::log_det_sympd(PG[i] / (this->sig2 * 2 * arma::datum::pi));
  }
  
  return 0.5 * (logdet - quad / this->sig2);
}



std::vector<arma::mat> VCMMModel::gradients(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  std::vector<arma::mat> R = this->residuals(Y, X, U);
  // std::vector<arma::colvec> R = this->global_residuals(Y, X, U, W, P);
  arma::mat grad_a(this->pu, 1);
  arma::mat grad_b(this->px, this->nt);
  grad_a.zeros();
  grad_b.zeros();
  double sw = 0.;
  
  for(uint i=0; i<Y.size(); i++){
    for(uint k=0; k<this->nt; k++){
      // arma::colvec rk = R[i];
      arma::colvec rk = R[i].col(k);
      arma::colvec wk = W[i].col(k);
      // sw += arma::accu(arma::sqrt(wk * wk.t()));
      arma::mat M = (P[i] % (arma::sqrt(wk * wk.t()))) * rk;
      grad_a += U[i].t() * M;
      grad_b.col(k) += X[i].t() * M;
    }
    sw += Y[i].n_elem;
  }
  // the objective is scaled by 1/n
  return std::vector<arma::mat>{- grad_a / sw, - grad_b / sw};
}



std::vector<std::vector<arma::mat>> VCMMModel::hessian_blocks(
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



arma::mat VCMMModel::hessian_from_blocks(
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
  std::vector<std::vector<arma::mat>> hessians = this->hessian_blocks(X, U, W, P);
  
  return this->hessian_from_blocks(
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
  std::vector<std::vector<arma::mat>> hessians = this->hessian_blocks(X, U, W, P);
  arma::mat hessian = this->hessian_from_blocks(
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
  
  // Heuristic might be a little small, increase until fine
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
  }
  
  this->La = La;
  this->Lb = Lb;
}

arma::mat VCMMModel::proximal_asgl(
  const arma::mat &b
){
  arma::mat m1 = this->lasso_weights * this->alpha * this->lambda / this->Lb;
  arma::colvec m2 = this->grplasso_weights * (1 - this->alpha) * this->lambda / this->Lb;
  return this->proximal_L1L2(b, m1, m2);
}

arma::rowvec VCMMModel::proximal_L1(
    const arma::rowvec &b,
    const arma::rowvec &m
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

arma::rowvec VCMMModel::proximal_L1L2_row(
    const arma::rowvec &b,
    const arma::rowvec &m1,
    const double m2
){
  // double m1 = this->alpha * this->lambda / this->Lb;
  // double m2 = sqrt(this->nt) * (1 - this->alpha) * this->lambda / this->Lb;
  return proximal_L2(proximal_L1(b, m1), m2);
}

arma::mat VCMMModel::proximal_L1L2(
    const arma::mat &b,
    const arma::mat &m1,
    const arma::colvec m2
){
  arma::mat out = b * 0.;
  for(uint j=0; j<this->px; j++){
    out.row(j) = this->proximal_L1L2_row(b.row(j), m1.row(j), m2[j]);
  }
  return out;
}


void VCMMModel::compute_ics(){
  this->bic = -2*this->mllk + this->bic_kernel;
  this->aic = -2*this->mllk + this->aic_kernel;
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

void VCMMModel::compute_df_kernel(
    const std::vector<arma::mat> & W, 
    double kernel_scale
){
  arma::rowvec n = this->effective_sample_size(W, kernel_scale);
  arma::rowvec df = this->active(); 
  // for now, we assume 1/T as the contribution of each time point, 
  // we also assume k(0)=1
  double range = this->t0.max() - this->t0.min();
  double scale = fmin(1, range / (this->nt * kernel_scale));
  this->bic_kernel = arma::dot(arma::log(n), df) * scale;
  this->aic_kernel = 2 * arma::accu(df) * scale;
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
    if(this->lambda > 0.) tmpb= this->proximal_asgl(tmpb);
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
  if(this->lambda > 0.) tmpb= this->proximal_asgl(tmpb);
  this->a = tmpa + (pt - 1.) * (tmpa - this->tmpa) / pt;
  this->b = tmpb + (pt - 1.) * (tmpb - this->tmpb) / pt;
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
  if(this->lambda > 0.) tmpb= this->proximal_asgl(tmpb);
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
  if(this->lambda > 0.) tmpb= this->proximal_asgl(tmpb);
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


void VCMMModel::re_ratio_nr_step_global(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P
){
  double deriv1 = 0.;
  double deriv2 = 0.;
  std::vector<arma::colvec> GR = this->global_residuals(Y, X, U, W, P);
  // std::vector<arma::mat> R = this->residuals(Y, X, U);
  std::vector<arma::mat> WG = this->global_weight(W);
  std::vector<arma::mat> PG = this->global_precision(W, P);
  
  for(uint i=0; i<Y.size(); i++){
    uint ni = Y[i].n_elem;
    double denum = ni*this->re_ratio + 1;
    double wrss = arma::dot(GR[i], WG[i] * GR[i]) ;
    // double wrss = 0.;
    // for(uint k=0; k<this->nt; k++){
    //   wrss += arma::dot(W[i].col(k), R[i].col(k));
    // }
    arma::mat PGinvWG = arma::inv(PG[i]) * WG[i];
    double Tr1 = arma::trace(PGinvWG);
    double Tr2 = arma::trace(PGinvWG * PGinvWG);
    deriv1 += -0.5 * Tr1 / pow(denum, 2);
    deriv1 += 0.5 * wrss / (pow(denum, 2) * this->sig2);
    deriv2 += ni * Tr1 / pow(denum, 3);
    deriv2 -= 0.5*Tr2 / pow(denum, 4);
    deriv2 -= ni * wrss / (pow(denum, 3) * this->sig2);
  }
  // Rcpp::Rcout << "[VCMM] deriv1=" << deriv1 << " deriv2=" << deriv2 <<  "\n";
  // Negative log NR step
  // return exp(deriv1/(deriv2*this->re_ratio+deriv1));
  // Negative NR step
  // return deriv1*0.01;
  if(deriv2 > 0.){
    this->re_ratio *= (deriv1 > 0.) ? 2. : 0.5; // jump somewhere else
  }else{
    double rho_step = -deriv1/deriv2;
    this->re_ratio += rho_step;
  }
  this->prev_re_ratio = this->re_ratio;
}

void VCMMModel::update_precision(
    std::vector<arma::mat> & P
){
  // This computes the unscaled Precision (without sig2)
  uint ni;
  for(uint i=0; i<P.size(); i++){
    uint ni = P[i].n_rows;
    P[i] = arma::eye(ni, ni);
    if(this->random_effect) P[i] -= 1/(ni + 1/this->re_ratio);
  }
}
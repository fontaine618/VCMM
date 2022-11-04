#include "RcppArmadillo.h"
#include "VCMMModel.hpp"
#include <math.h>

// [[Rcpp::depends(RcppArmadillo)]]

VCMMModel::VCMMModel(
  const int px,
  const int pu,
  const int nt,
  const int q,
  const double alpha,
  const double lambda
){
  this->px = px;
  this->pu = pu;
  this->nt = nt;
  this->q = q;
  this->alpha = alpha;
  this->lambda = lambda;

  // initialize coefficients to 0
  arma::mat b(px, nt);
  b.zeros();
  arma::mat a(pu, 1);
  a.zeros();
  this->a = a;
  this->b = b;
}

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
      sw += arma::accu(wk * wk.t());
    }
  }

  return 0.5 * loss / sw;
}

double VCMMModel::penalty(){
  double penalty;
  double l1, l2;

  l1 = sum(abs(this->b.row(1)));
  l2 = sum(square(this->b.row(1)));
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
      sw += arma::accu(wk * wk.t());
      arma::mat M = (P[i] % (arma::sqrt(wk * wk.t()))) * rk;
      grad_a += U[i].t() * M;
      grad_b.col(k) += X[i].t() * M;
    }
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
  Rcpp::Rcout << "1\n";
  
  // initialize to 0
  hessian_a.zeros();
  for(uint i=0; i<this->nt; i++){
    hessian_b[i] = arma::zeros(this->px, this->px);
    hessian_ab[i] = arma::zeros(this->px, this->pu);
  }
  Rcpp::Rcout << "2\n";
  
  // compute hessian
  for(uint i=0; i<X.size(); i++){
    for(uint k=0; k<this->nt; k++){
      arma::colvec wk = W[i].col(k);
      sw += arma::accu(wk * wk.t());
      arma::mat M = P[i] % (arma::sqrt(wk * wk.t()));
      hessian_a += U[i].t() * M * U[i];
      hessian_b[k] += X[i].t() * M * X[i];
      hessian_ab[k] += X[i].t() * M * U[i];
    }
  }
  Rcpp::Rcout << "3\n";
  
  // divide by total weight
  hessian_a /= sw;
  for(uint i=0; i<this->nt; i++){
    hessian_b[i] /= sw;
    hessian_ab[i] /= sw;
  }
  Rcpp::Rcout << "4\n";
  
  
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
  Rcpp::Rcout << "5\n";
  uint dim = this->pu + this->nt * this->px;
  arma::mat hessian(dim, dim);
  hessian.zeros();
  
  Rcpp::Rcout << "6\n";
  hessian.submat(0, 0, this->pu - 1, this->pu - 1) = hessian_a;
  for(uint k=0; k<this->nt; k++){
    hessian.submat(0,
                   this->pu + k*this->px,
                   this->pu - 1,
                   this->pu + (k+1)*this->px - 1) = hessian_ab[k].t();
    hessian.submat(this->pu + k*this->px,
                   0,
                   this->pu + (k+1)*this->px - 1,
                   this->pu - 1) = hessian_ab[k];
    hessian.submat(this->pu + k*this->px,
                   this->pu + k*this->px,
                   this->pu + (k+1)*this->px - 1,
                   this->pu + (k+1)*this->px - 1) = hessian_b[k];
  }
  Rcpp::Rcout << "7\n";
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
  
  arma::eig_sym(eigval, hessians[0][0]);
  La = eigval.max();
  
  Lb = 0;
  for(uint k=0; k<this->nt; k++){
    arma::eig_sym(eigval, hessians[1][k]);
    Lb = fmax(Lb, eigval.max());
  }
  
  Rcpp::Rcout << "Updating La and Lb until H - D is positive definite \n";
  Rcpp::Rcout << "La: " << La << " Lb: " << Lb << "\n";
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
  Rcpp::Rcout << "0 minevalHmLI: " << minevalHmLI << "\n";
  
  uint iter=0;
  while(minevalHmLI<=0.){
    iter++;
    La *= 1.1;
    Lb *= 1.1;
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
    Rcpp::Rcout << iter << " minevalHmLI: " << minevalHmLI << "\n";
  }
  
  this->La = La;
  this->Lb = Lb;
  Rcpp::Rcout << "La: " << this->La << " Lb: " << this->Lb << "\n";
}

arma::rowvec _proximal_L1(
    const arma::rowvec &b,
    double m
){
  uint nt = b.n_elem;
  arma::rowvec s(nt);
  s = arma::abs(b) - m;
  s = arma::sign(b) % arma::clamp(s, 0., arma::datum::inf);
  return s;
}

arma::rowvec _proximal_L2(
    const arma::rowvec &s,
    double m
){
  double sn = arma::norm(s);
  if(sn == 0.) sn = 1.; // to avoid dividing by 0, the value doesn't matter since s=0 in that case
  return fmax(1. - m/sn, 0.) * s;
}

arma::rowvec VCMMModel::proximal(
    const arma::rowvec &b
){
  double m1 = this->alpha * this->lambda * this->Lb;
  double m2 = sqrt(this->nt) * (1 - this->alpha) * this->lambda * this->Lb;
  return _proximal_L2(_proximal_L1(b, m1), m2);
}

// FISTA step
void VCMMModel::step(
    const std::vector<arma::mat> &gradients
){
  // compute regular update using L
  arma::colvec new_a = this->a - gradients[0] / this->La;
  arma::mat new_b = this->b - gradients[1] / this->Lb;
  new_b.row(1) = this->proximal(new_b.row(1));
  // update stepsize and take momentum step
  double pt = this->momentum;
  double momentum = 0.5 * (1. + sqrt(1 + 4.*pt*pt));
  this->a = new_a + (pt - 1.) * (new_a - this->a) / momentum;
  this->b = new_b + (pt - 1.) * (new_b - this->b) / momentum;
  this->momentum = momentum;
}

double VCMMModel::compute_lambda_max(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P,
    uint max_iter,
    double rel_tol
){
  this->lambda = 1e6;
  this->fit(Y, X, U, W, P, max_iter, rel_tol);
  
  std::vector<arma::mat> gradients = this->gradients(Y, X, U, W, P);
  arma::mat gb1 = gradients[1].row(1);
  arma::mat b1 = -gb1 / this->Lb;
  double b1norm2 = arma::norm(b1, "fro");
  double b1norminf = arma::norm(b1, "inf");
  double lambda_max = -1e10;
  // L1 bound
  if(this->alpha > 0.){
    lambda_max = fmax(lambda_max, b1norminf / (this->Lb * this->alpha));
  }
  // L2 bound
  if(this->alpha < 1.){
    lambda_max = fmax(lambda_max, b1norm2 / (this->Lb * sqrt(this->nt) * (1. - this->alpha)));
  }
  // decrease until no longer 0
  double m1 = this->alpha * lambda_max * this->Lb;
  double m2 = sqrt(this->nt) * (1 - this->alpha) * lambda_max * this->Lb;
  arma::mat b = _proximal_L2(_proximal_L1(b1, m1), m2);
  while(arma::norm(b, "inf") < 1.e-10){
    lambda_max *= 0.95;
    m1 = this->alpha * lambda_max * this->Lb;
    m2 = sqrt(this->nt) * (1 - this->alpha) * lambda_max * this->Lb;
    b = _proximal_L2(_proximal_L1(b1, m1), m2);
  }
  
  return lambda_max / 0.95; // we went one iteration too far
}

double VCMMModel::fit(
    const std::vector<arma::colvec> & Y,
    const std::vector<arma::mat> & X,
    const std::vector<arma::mat> & U,
    const std::vector<arma::mat> & W,
    const std::vector<arma::mat> & P,
    uint max_iter,
    double rel_tol
){
  std::vector<arma::mat> g;
  this->momentum = 1.; // reset momentum
  double prev_loss = this->loss(Y, X, U, W, P);
  for(uint iter=1; iter<=max_iter; iter++){
    g = this->gradients(Y, X, U, W, P);
    this->step(g);
    double loss = this->loss(Y, X, U, W, P);
    double rel_change = 2 * fabs(prev_loss - loss) / fabs(prev_loss + loss);
    if(iter % 1 == 0) Rcpp::Rcout << 
      "Iteration " << iter << 
        " Loss: " << loss << 
          " momentum: " << this->momentum << 
            " rel_change: " << rel_change << "\n";
    prev_loss = loss;
    if(rel_change < rel_tol) break;
  }
  return prev_loss;
}

// TODO: max_iter and rel_tol as members
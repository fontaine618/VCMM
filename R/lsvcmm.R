#' Title
#'
#' @param response 
#' @param subject 
#' @param time 
#' @param vc_covariates 
#' @param nvc_covariates 
#' @param data 
#' @param random_design 
#' @param vc_intercept 
#' @param estimated_time 
#' @param sgl 
#' @param lambda 
#' @param lambda_factor 
#' @param n_lambda 
#' @param adaptive 
#' @param kernel 
#' @param kernel_scale 
#' @param kernel_scale_factor 
#' @param n_kernel_scale 
#' @param tuning_strategy 
#' @param ebic_factor 
#' @param cv 
#' @param cv_seed 
#' @param control 
#'
#' @return
#' @export
#'
#' @examples
lsvcmm = function(
  # data
  response,
  subject,
  time,
  vc_covariates=NULL,
  nvc_covariates=NULL,
  data=NULL,
  
  # design
  random_effect=T,
  estimate_variance_components=F,
  re_ratio=-1,
  vc_intercept=TRUE,
  estimated_time=NULL,
  
  # regularization
  sgl=1.,
  lambda=NULL,
  lambda_factor=0.005,
  n_lambda=100,
  adaptive=0.,
  
  # smoothing
  kernel=c("squared_exponential"),
  kernel_scale=NULL,
  kernel_scale_factor=10,
  n_kernel_scale=20,
  
  # tuning
  tuning_strategy=c("grid_search"),
  
  # cross-validation
  cv=NULL,
  cv_seed=0,
  
  # parameters
  control=lsvcmm_control()
){
  # PARAMETERS
  if(!inherits(control, "lsvcmm_control")){
    if(!is.list(control)) stop("control must either be a lsvcmm_control object or a list")
    control = do.call(lsvcmm_control, control)
  }
  
  # DATA PREPARATION
  d = prepare_data(response, subject, time, vc_covariates, nvc_covariates, data, vc_intercept)
  
  tt = prepare_time(d$t, estimated_time, control[["scale_time"]])
  
  # REGULARIZATION PREPARATION
  stopifnot((sgl >= 0) && (sgl <=1))
  if(is.null(lambda)){
    lambda = numeric(0L)
    stopifnot(lambda_factor>0)
    stopifnot(n_lambda>0)
  }else{
    stopifnot(min(lambda)>=0)
    lambda_factor = 0
    n_lambda = 1
  }
  stopifnot(adaptive>=0.)
  
  # KERNEL PREPARATION
  kernel = match.arg(kernel)
  if(is.null(kernel_scale) || kernel_scale==0){
    kernel_scale = numeric(0L)
    stopifnot(kernel_scale_factor>1)
    stopifnot(n_kernel_scale>0)
  }else{
    stopifnot(min(kernel_scale)>=0)
    kernel_scale_factor = 1
    n_kernel_scale = 1
  }
  
  # TUNING PREPARATION
  tuning_strategy = match.arg(tuning_strategy)

  # TUNING PREPARATION
  if(is.null(cv)) cv = 0
  if(cv == -1) cv = length(d$subject_ids)  # LOO
  
  # CALL
  obj = VCMM(
    response=d$y, 
    subject=d$s, 
    response_time=tt$t,
    vcm_covariates=d$X,
    fixed_covariates=d$U,
    estimated_time=tt$t0,
    random_effect=random_effect,
    estimate_variance_components=estimate_variance_components,
    re_ratio=re_ratio,
    tuning_strategy=tuning_strategy,
    kernel_scale=kernel_scale, 
    kernel_scale_factor=kernel_scale_factor,
    n_kernel_scale=n_kernel_scale,
    alpha=sgl,
    lambda=lambda,
    lambda_factor=lambda_factor,
    n_lambda=n_lambda, 
    adaptive=adaptive,
    penalize_intercept=!vc_intercept,
    max_iter=control[["max_iter"]],
    rel_tol=control[["rel_tol"]],
    nfolds=cv,
    cv_seed=cv_seed,
    progress_bar=control[["progress_bar"]]
  )
  
  # SUMMARIZE RESULTS
  models_path = data.frame(
    kernel_scale=sapply(obj$models, function(model) model$kernel_scale),
    lambda=sapply(obj$models, function(model) model$lambda),
    df_vc=sapply(obj$models, function(model) model$df_vc),
    bic_kernel=sapply(obj$models, function(model) model$bic_kernel),
    aic_kernel=sapply(obj$models, function(model) model$aic_kernel),
    objective=sapply(obj$models, function(model) model$objective),
    penalty=sapply(obj$models, function(model) model$penalty),
    rss=sapply(obj$models, function(model) model$rss),
    parss=sapply(obj$models, function(model) model$parss),
    sig2=sapply(obj$models, function(model) model$sig2),
    re_ratio=sapply(obj$models, function(model) model$re_ratio),
    mllk=sapply(obj$models, function(model) model$mllk),
    aic=sapply(obj$models, function(model) model$aic),
    bic=sapply(obj$models, function(model) model$bic),
    cv_score=sapply(obj$models, function(model) model$cv_score)
  )
  best_idx = which.min(models_path$aic)
  
  # AGGREGATE COEFFICIENTS
  a = NULL
  if(d$pu>0) a = matrix(sapply(obj$models, function(model) model$a, simplify="matrix"), nrow=d$pu)
  b = sapply(obj$models, function(model) model$b, simplify="array")
  
  # RESULTS PREPARATION
  res = list(
    sgl=sgl,
    random_effect=random_effect,
    vc_intercept=vc_intercept,
    t_range=tt$t_range,
    estimated_time=tt$t0*diff(tt$t_range) + tt$t_range[1],
    control=control,
    tuning_strategy=tuning_strategy,
    models_path=models_path,
    best_idx=best_idx,
    best_ebic=models_path$ebic[best_idx],
    best_lambda=models_path$lambda[best_idx],
    best_kernel_scale=models_path$kernel_scale[best_idx],
    best_model=obj$models[[best_idx]],
    best_nvc=a[, best_idx],
    best_vc=b[,,best_idx],
    nvc_path=a,
    vc_path=b,
    cv_folds=cv
  )
  
  class(res) = c("vcmm")
  return(res)
}

# TODO: need to store variable names
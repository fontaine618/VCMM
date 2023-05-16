lsvcmm.boot = function(
  response,
  subject,
  time,
  vc_covariates=NULL,
  nvc_covariates=NULL,
  data=NULL,
  
  # design
  random_effect=T,
  estimate_variance_components=F,
  vc_intercept=TRUE,
  estimated_time=NULL,
  
  # regularization
  sgl=1.,
  lambda=0.,
  adaptive=0.,
  
  # smoothing
  kernel=c("squared_exponential"),
  kernel_scale=1.,
  
  # bootstrap
  n_samples=1000,
  
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
  stopifnot(lambda>=0.)
  stopifnot(adaptive>=0.)
  
  # KERNEL PREPARATION
  kernel = match.arg(kernel)
  stopifnot(kernel_scale>0)
  
  # CALL
  obj = VCMMBoot(
    response=d$y, 
    subject=d$s, 
    response_time=tt$t,
    vcm_covariates=d$X,
    fixed_covariates=d$U,
    estimated_time=tt$t0,
    random_effect=random_effect,
    estimate_variance_components=estimate_variance_components,
    kernel_scale=kernel_scale, 
    alpha=sgl,
    lambda=lambda,
    adaptive=adaptive,
    penalize_intercept=!vc_intercept,
    max_iter=control[["max_iter"]],
    rel_tol=control[["rel_tol"]],
    n_samples=n_samples,
    progress_bar=control[["progress_bar"]]
  )
  
  # AGGREGATE COEFFICIENTS
  a = NULL
  if(d$pu>0) a = matrix(sapply(obj$boot, function(model) model$a, simplify="matrix"), nrow=d$pu)
  b = sapply(obj$boot, function(model) model$b, simplify="array")
  
  
  out = list(
    nvc_boot=a,
    vc_boot=b,
    nvc=obj$model$a,
    vc=obj$model$b,
    estimated_time=tt$t0*diff(tt$t_range) + tt$t_range[1]
  )
  class(out) = c("VCMMBoot")
  return(out)
}

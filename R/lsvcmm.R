lsvcmm = function(
  # data
  response,
  subject,
  time,
  vc_covariates=NULL,
  nvc_covariates=NULL,
  data=NULL,
  
  # design
  random_design=c("intercept"),
  vc_intercept=TRUE,
  estimated_time=NULL,
  
  # regularization
  sgl=1.,
  lambda=NULL,
  lambda_factor=0.005,
  n_lambda=100,
  
  # smoothing
  kernel=c("squared_exponential"),
  kernel_scale=NULL,
  kernel_scale_factor=10,
  n_kernel_scale=20,
  
  # tuning
  tuning_strategy=c("grid_search", "orthogonal_search", "bisection"),
  ebic_factor=1.,
  
  # cross-validation
  cv=NULL,
  
  # parameters
  control=lsvcmm_control()
){
  # PARAMETERS
  if(!inherits(control, "lsvcmm_control")){
    if(!is.list(control)) stop("control must either be a lsvcmm_control object or a list")
    control = do.call(lsvcmm_control, control)
  }
  
  # DATA PREPARATION
  if(!is.null(data)){ # here, we expect response, time, nvc_covariates and vc_covariates to be strings/indices
    response = data[[response]]
    subject = data[[subject]]
    time = data[[time]]
    if(!is.null(vc_covariates)){
      if(is.character(vc_covariates)) vc_covariates = matrix(data[[vc_covariates]], ncol=1)
      if(is.vector(vc_covariates)) vc_covariates = as.matrix(subset(data, select=vc_covariates))
    }
    if(!is.null(nvc_covariates)){
      nvc_covariates = as.matrix(subset(data, select=nvc_covariates))
    }else{
      nvc_covariates = matrix(numeric(0L), nrow=nrow(vc_covariates), ncol=0)
    }
  } # we assume all terms to be already in vector/matrix form here on
  n = length(response)
  y = matrix(response, n, 1)
  subject_ids = sort(unique(subject))
  s = sapply(subject, function(ss) which.max(ss==subject_ids)) - 1
  s = matrix(s, n, 1)
  t = matrix(time, n, 1)
  if(!is.null(vc_covariates)){
    X = vc_covariates
    if(vc_intercept) X = cbind(1, X)
  }else{
    if(vc_intercept){
      X = matrix(1, n, 1)
    }else{stop("either vc_covariates or vc_intercept must be specified (there are no vc otherwise!)")}
  }
  U = nvc_covariates
  
  px = ncol(X)
  pu = ncol(U)
  
  # RESCALE TIME
  if(control[["scale_time"]]){
    t_range = range(t)
    if(!is.null(estimated_time)){
      t_range[1] = min(min(estimated_time), t_range[1]) 
      t_range[2] = max(max(estimated_time), t_range[2]) 
    }
    t = (t - t_range[1]) / diff(t_range)
  }
  
  # TIME POINTS
  if(is.null(estimated_time)){
    nt = length(unique(t))
    if(nt * px > n) {
      nt = floor(n / px)
      estimated_time = unname(quantile(t, seq(0, nt-1)/(nt-1)))
    }else{
      estimated_time = unique(t)
    }
  }else{
    if(control[["scale_time"]]){
      estimated_time = (estimated_time - t_range[1]) / diff(t_range)
    }
  }
  t0 = sort(unique(estimated_time))
  
  # RANDOM DESIGN
  random_design = match.arg(random_design)
  if(random_design=="intercept"){
    Z = matrix(1, n, 1)
  }
  
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
  
  # KERNEL PREPARATION
  kernel = match.arg(kernel)
  if(is.null(kernel_scale)){
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
  stopifnot(ebic_factor>=0)
  
  # TUNING PREPARATION
  if(is.null(cv)) cv = 0
  if(cv == -1) cv = length(subject_ids) - 1  # LOO
  
  
  # CALL
  obj = VCMM(
    response=y, 
    subject=s, 
    response_time=t,
    random_design=Z,
    vcm_covariates=X,
    fixed_covariates=U,
    estimated_time=t0,
    tuning_strategy=tuning_strategy,
    kernel_scale=kernel_scale, 
    kernel_scale_factor=kernel_scale_factor,
    n_kernel_scale=n_kernel_scale,
    alpha=sgl,
    lambda=lambda,
    lambda_factor=lambda_factor,
    n_lambda=n_lambda, 
    max_iter=control[["max_iter"]],
    mult=-1,
    ebic_factor=ebic_factor,
    rel_tol=control[["rel_tol"]],
    orthogonal_search_max_rounds=control[["orthogonal_search_max_rounds"]],
    bissection_max_evals=control[["bissection_max_evals"]],
    nfolds=cv
  )
  
  # SUMMARIZE RESULTS
  models_path = data.frame(
    kernel_scale=sapply(obj$models, function(model) model$kernel_scale),
    lambda=sapply(obj$models, function(model) model$lambda),
    df_vc=sapply(obj$models, function(model) model$df_vc),
    df_kernel=sapply(obj$models, function(model) model$df_kernel),
    objective=sapply(obj$models, function(model) model$objective),
    penalty=sapply(obj$models, function(model) model$penalty),
    rss=sapply(obj$models, function(model) model$rss),
    parss=sapply(obj$models, function(model) model$parss),
    sig2=sapply(obj$models, function(model) model$sig2),
    apllk=sapply(obj$models, function(model) model$apllk),
    amllk=sapply(obj$models, function(model) model$amllk),
    bic=sapply(obj$models, function(model) model$bic),
    ebic=sapply(obj$models, function(model) model$ebic),
    predparss=sapply(obj$models, function(model) model$predparss)
  )
  best_idx = which.min(models_path$ebic)
  
  # AGGREGATE COEFFICIENTS
  a = NULL
  if(pu>0) a = matrix(sapply(obj$models, function(model) model$a, simplify="matrix"), nrow=pu)
  b = sapply(obj$models, function(model) model$b, simplify="array")
  
  # RESULTS PREPARATION
  res = list(
    sgl=sgl,
    random_design=random_design,
    vc_intercept=vc_intercept,
    t_range=t_range,
    estimated_time=t0*diff(t_range) + t_range[1],
    ebic_factor=ebic_factor,
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
    vc_path=b
  )
  
  class(res) = c("vcmm")
  return(res)
}
prepare_data = function(
  response,
  subject,
  time,
  vc_covariates=NULL,
  nvc_covariates=NULL,
  data=NULL,
  random_design=c("intercept"),
  vc_intercept=TRUE
){
  if(!is.null(data)){ # here, we expect response, time, nvc_covariates and vc_covariates to be strings/indices
    response = data[[response]]
    subject = data[[subject]]
    time = data[[time]]
    if(!is.null(vc_covariates)){
       vc_covariates = as.matrix(subset(data, select=vc_covariates))
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
  
  # RANDOM DESIGN
  random_design = match.arg(random_design)
  if(random_design=="intercept"){
    Z = matrix(1, n, 1)
  }
  
  return(list(
    t=t,
    n=n,
    y=y,
    s=s,
    subject_ids=subject_ids,
    X=X,
    U=U,
    px=px,
    pu=pu,
    Z=Z
  ))
}


prepare_time = function(
  t,
  estimated_time, 
  scale_time
){
  # RESCALE TIME
  if(scale_time){
    t_range = range(t)
    if(!is.null(estimated_time)){
      t_range[1] = min(min(estimated_time), t_range[1]) 
      t_range[2] = max(max(estimated_time), t_range[2]) 
    }
    t = (t - t_range[1]) / diff(t_range)
  }else{
    t_range=NULL
  }
  
  # TIME POINTS
  if(is.null(estimated_time)){
    nt = length(unique(t))
    if(nt > 100) {
      estimated_time = unname(quantile(t, seq(0, nt-1)/(nt-1)))
    }else{
      estimated_time = unique(t)
    }
  }else{
    if(scale_time){
      estimated_time = (estimated_time - t_range[1]) / diff(t_range)
    }
  }
  t0 = sort(unique(estimated_time))
  return(list(
    t0=t0,
    nt=length(t0),
    t=t,
    t_range=t_range
  ))
}
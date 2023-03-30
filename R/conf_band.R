confidence_band = function(
  obj,
  level=0.95,
  method=c("quantile"),
  var=2
){
  if(!("VCMMBoot" %in% class(obj))) stop("obj must be of class VCMMBoot, i.e., the outpute of a lsvcmm.boot call")
  stopifnot(0<level & level<1)
  level = 1-level
  method = match.arg(method)
  samples = obj$vc_boot[var, , ]
  
  out = switch(
    method,
    "quantile"=quantile_confidence_band(samples, level)
  )
  
  out = data.frame(
    L=out$L, 
    U=out$U,
    pointwise_confidence=1-2*out$p,
    estimate=obj$vc[var, ],
    mean=apply(samples, 1, mean),
    median=apply(samples, 1, median),
    prop0=apply(samples, 1, function(x) mean(x!=0)),
    estimated_time=obj$estimated_time,
    level=proportion_samples_in_band(out$L, out$U, samples),
    excludes_zero=(out$L>0) | (out$U<0)
  )
  
  return(out)
}


quantile_confidence_band = function(samples, level){
  nt = dim(samples)[1]
  minp = level / (2*nt)
  maxp = level / 2
  ps = pracma::logseq(minp, maxp, 200)
  prop = sapply(ps, function(p){
    out = pointwise_quantile_confidence_band(p, samples)
    proportion_samples_in_band(out$L, out$U, samples)
  })
  valid = prop >= 1-level
  which = length(valid) + 1 - which.max(rev(valid))
  p = ps[which]
  out = pointwise_quantile_confidence_band(p, samples)
  return(out)
}


pointwise_quantile_confidence_band = function(p, samples){
  qs = apply(samples, 1, quantile, c(p, 1-p))
  L = qs[1, ]
  U = qs[2, ]
  return(list(L=L, U=U, p=p))
}

proportion_samples_in_band = function(L, U, samples){
  aL = apply(samples, 2, function(x) all(x>=L))
  bU = apply(samples, 2, function(x) all(x<=U))
  between = aL*bU
  return(mean(between))
}

pointwise_pvalues = function(samples){
  B = dim(samples)[2]
  pos = apply(samples, 1, function(x) sum(x>0))
  neg = apply(samples, 1, function(x) sum(x<0))
  ppos = 1-pos/B
  pneg = 1-neg/B
  pval_npm = pmin(2 * pmin(ppos, pneg), 1)
  mus = apply(samples, 1, mean)
  sds = apply(samples, 1, sd)
  pval_norm = 2 * pnorm(abs(mus)/sds, lower.tail=F)
  return(list(npm=pval_npm, norm=pval_norm))
}

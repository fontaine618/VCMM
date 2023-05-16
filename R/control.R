lsvcmm_control = function(
  max_iter=1000,
  rel_tol=1e-6,
  progress_bar=T,
  scale_time=T
){
  ctrl = list(
    max_iter=max_iter,
    rel_tol=rel_tol,
    progress_bar=progress_bar,
    scale_time=scale_time
  )
  class(ctrl) = "lsvcmm_control"
  return(ctrl)
}
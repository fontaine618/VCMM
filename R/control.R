lsvcmm_control = function(
  max_iter=1000,
  rel_tol=1e-6,
  progress_bar=T,
  orthogonal_search_max_rounds=2,
  bissection_max_evals=20,
  scale_time=T
){
  ctrl = list(
    max_iter=max_iter,
    rel_tol=rel_tol,
    progress_bar=progress_bar,
    orthogonal_search_max_rounds=orthogonal_search_max_rounds,
    bissection_max_evals=bissection_max_evals,
    scale_time=scale_time
  )
  class(ctrl) = "lsvcmm_control"
  return(ctrl)
}
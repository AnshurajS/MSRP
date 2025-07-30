# forward_wrapper.py
import numpy as np
from .prt_forward import forward_model
from .data_obs    import w_trans, w_emit
from petitRADTRANS import physical_constants as cst

def run_forward(theta):
    """
    ? = 4-vector of log-abundances.
    Returns a 1D spectrum of length len(w_trans)+len(w_emit),
    interpolated onto your retrieval grid.
    """
    wl_t, depth_t, wl_e, flux_e = forward_model(theta)

    # convert from cm?¹ ? meu-m
    c = cst.c
    lam_t = c / (wl_t * 1e2)
    lam_e = c / (wl_e * 1e2)

    trans_obs = np.interp(w_trans, lam_t, depth_t)
    emit_obs  = np.interp(w_emit, lam_e, flux_e)

    return np.concatenate([trans_obs, emit_obs])

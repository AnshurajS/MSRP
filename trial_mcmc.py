# -*- coding: utf-8 -*-

import os, emcee, corner, numpy as np
import multiprocessing as mp
from .prt_forward import forward_model, planet
from .data_obs import w_trans, w_emit, e_trans, e_emit
from .forward_wrapper import cst as c
import matplotlib.pyplot as plt

# -- 1. Define priors & likelihood for the two parameters --------------------
def log_prior(theta):
    logX_H2O, logX_CH4, logX_CO, logX_CO2 = theta
    if not (-12 < logX_H2O < -1 and -12 < logX_CH4 < -1 and -12<logX_CO<-1 and -12<logX_CO2<-1):
        return -np.inf
    return 0.0

def log_likelihood(theta):
    wl_t, mod_t, wl_e, mod_e = forward_model(theta)
    # interpolate model onto observed grids
    mt = np.interp(w_trans, wl_t, mod_t)
    me = np.interp(w_emit, wl_e, mod_e)
    lnL_t = -0.5 * np.sum(((w_trans - mt)/e_trans)**2)
    lnL_e = -0.5 * np.sum(((w_emit  - me)/e_emit )**2)
    return lnL_t + lnL_e

def log_posterior(theta):
    lp = log_prior(theta)
    return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

def main():
    # -- 2. Run emcee -------------------------------------------------------------
    ndim, nwalkers = 4, 32
    # start near plausible values
    p0 = np.array([-4.0, -6.0, -5.0, -5.0])
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    pool = mp.Pool(32)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
    sampler.run_mcmc(pos, 50, progress=True)

    pool.close()
    pool.join()

    # -- 3. Burn-in, flatten & corner-plot ----------------------------------------
    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(2*np.max(tau))
    thin   = max(1, int(0.2*np.min(tau)))
    print(f"Using burn-in = {burnin} steps, thin = {thin}")

    labels = ["logX_H2O","logX_CH4","logX_CO","logX_CO2"]
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    fig = corner.corner(
        samples, labels=labels,
        truths=None, show_titles=True, title_fmt=".2f"
    )

    # save
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/corner_4param_32c_500itr.png", dpi=200)
    print("Corner plot saved to results/corner_4params.png")

    # -- 4. Best-fit model & data plots -------------------------------------------
    # median parameter set
    theta_med = np.median(samples, axis=0)
    wl_t, depth_t, wl_e, flux_e = forward_model(theta_med)

    # Transmission plot
    plt.figure()
    plt.errorbar(w_trans, e_trans, yerr=e_trans, fmt='o', label='Data')
    R_star = planet.star_radius
    depth_model = (depth_t/R_star)**2
    plt.plot(wl_t, depth_model, '-', label='Model (median)')
    plt.xlabel('Wavelength [meu-m]')
    plt.ylabel('Transit depth')
    plt.legend()
    plt.title('Transmission Fit')
    plt.savefig("results/transmission_fit_revised.png", dpi=200)

    # Emission plot
    plt.figure()
    plt.errorbar(w_emit, e_emit, yerr=e_emit, fmt='o', label='Data')
    # 1) Wavelength in meters
    lam_m = wl_e * 1e-6

    # 2) Planck function B_lambda [W m^-2 sr^-1 m^-1]
    h, ck, kB = c.h, c.c, c.kB
    exponent = h * ck / (lam_m * kB * planet.star_effective_temperature)
    denom = np.expm1(exponent)
    B_lambda = np.where(
        exponent < 700,
        (2*h*ck**2) / (lam_m**5) / denom,
        0.0  # Avoid overflow for large exponents
    )

    # 3) Stellar flux at Earth: p * B_lambda * (R_star / d)^2
    #   (planet.radius_star and planet.system_distance must be in meters)
    F_star = np.pi * B_lambda * (planet.star_radius / planet.system_distance)**2

    # 4) Planet/star flux ratio
    flux_ratio = np.where(
        F_star > 0,
        flux_e / F_star,
        np.nan  # Avoid division by zero
    )
    plt.plot(wl_e, flux_ratio, '-', label='Model (median)')
    plt.xlabel('Wavelength [meu-m]')
    plt.ylabel('Flux ratio')
    plt.legend()
    plt.title('Emission Fit')
    plt.savefig("results/emission_fit_revised.png", dpi=200)

    print("Saved: results/corner_4param.png, transmission_fit.png, emission_fit.png")
if __name__ == "__main__":
    main()

# prt_forward.py
import numpy as np
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.planet import Planet

# 1) shared setup (runs on import but is cheap relative to each call)
planet    = Planet.get('HD 189733 b')
pressures = np.logspace(-6,2,50)

radtrans = Radtrans(
    line_species             = ["H2O","CH4","CO-NatAbund","CO2"],
    gas_continuum_contributors = ["H2--H2","H2--He"],
    pressures                = pressures
)

MW = {  # g/mol
    "H2O":18.01528, "CH4":16.04,
    "CO-NatAbund":28.01, "CO2":44.01,
    "H2":2.01588,     "He":4.002602
}

def forward_model(theta: np.ndarray):
    """
    ? = [logX_H2O, logX_CH4, logX_CO, logX_CO2]
    Returns: four arrays (wl_t, depth_t, wl_e, flux_e) on pRT's native grid.
    """
    logX_H2O, logX_CH4, logX_CO, logX_CO2 = theta
    X = {
      "H2O":10**logX_H2O, "CH4":10**logX_CH4,
      "CO-NatAbund":10**logX_CO, "CO2":10**logX_CO2,
      "He":0.15
    }
    X["H2"] = 1.0 - sum(X.values())

    # mass fractions per layer
    denom = sum(X[sp]*MW[sp] for sp in X)
    mfracs = {sp: np.full_like(pressures, (X[sp]*MW[sp]/denom))
              for sp in X}

    # isothermal T
    T_iso = 1200.0 * np.ones_like(pressures)
    grav  = planet.reference_gravity
    R_pl  = planet.radius
    p0    = planet.reference_pressure

    # transit
    wl_t, depth_t, _ = radtrans.calculate_transit_radii(
      temperatures       = T_iso,
      mass_fractions     = mfracs,
      mean_molar_masses  = 1.0/ sum(X[sp]/MW[sp] for sp in X),
      reference_gravity  = grav,
      planet_radius      = R_pl,
      reference_pressure = p0
    )

    # emission
    wl_e, flux_e, _ = radtrans.calculate_flux(
      temperatures       = T_iso,
      mass_fractions     = mfracs,
      mean_molar_masses  = 1.0/ sum(X[sp]/MW[sp] for sp in X),
      reference_gravity  = grav,
      planet_radius      = R_pl
    )

    return wl_t, depth_t, wl_e, flux_e

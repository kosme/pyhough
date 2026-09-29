import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    constants: dict


def constants():
    # fundamental constants
    c = 299792458  # speed of light
    h = 6.626e-34  # Planck's constant
    hbar = h / (2 * np.pi)
    ev = 1.602e-19  # electron volt
    Msun = 1.9885e30  # solar mass
    G = 6.67430e-11  # gravitational constant
    f_earth = 1 / 86400  # Earth's rotation frequency
    # length of a stellar day in seconds
    StellarDay = 23 * 3600 + 56 * 60 + 4.098903691

    v0 = 0.000766667  # over c
    vesc = 0.00181459  # over c
    rhodm = 0.4e9 / 1e-6  # dark matter energy density (eV/m3)

    # Earth's orbital angular frequency
    omega_earth_orb = 2 * np.pi / (365.25 * 86400)
    omega_earth_rot = 2 * np.pi * f_earth  # Earth's rotational angular frequency
    R_earth = 6371e3  # Earth's radius in meters
    Rorb = 149597871e3  # Earth's orbital radius in meters
    eps0 = 8.854187e-12  # vacuum permittivity
    fine_struct = 1 / 137  # fine structure constant

    units = {
        'ev_to_inv_s': ev / hbar,
        'ev_to_inv_m': ev / (hbar * c),
        'ev_to_kg': ev / c**2,
        'kg_to_ev': c**2 / ev,
        'charge_LH': ev / np.sqrt(4 * np.pi * fine_struct),
        'charge_G': ev / np.sqrt(fine_struct),
        'kpc_to_m': 3.086e19,
        'mpc_to_m': 3.086e22
    }

    consts = {
        'c': c,
        'h': h,
        'hbar': hbar,
        'hbar_inev': hbar / ev,
        'ev': ev,
        'Msun': Msun,
        'G': G,
        'eps0': eps0,
        'fine_struct': fine_struct,
        'f_earth': f_earth,
        'omega_earth_rot': omega_earth_rot,
        'omega_earth_orb': omega_earth_orb,
        'v_earth_rot': omega_earth_rot * R_earth,
        'v_earth_orb': omega_earth_orb * Rorb,
        'Rorb': Rorb,
        'Re': R_earth,
        'v0': v0,
        'vesc': vesc,
        'rhodm': rhodm,
        'StellarDay': StellarDay,
        'units': units
    }

    return Constants(consts).constants

# Example usage:
# constants_data = constants()
# print(constants_data)


def calc_mc_with_k(k):
    """
    Compute chirp mass in solar masses from the chirp mass factor.

    Parameters
    ----------
    k : float or ndarray
        k parameter(s)

    Returns
    -------
    mc : float or ndarray
        Chirp mass in solar masses.
    """
    if not isinstance(k, (float, np.ndarray)):
        raise TypeError("Input must be float or ndarray")

    if np.any(k <= 0) or np.any(np.isnan(k)):
        raise ValueError("k must be a positive, non-zero value.")

    # Assuming constants() is a function that returns the constants dictionary
    consts = constants()
    G = consts['G']
    c = consts['c']
    msun = consts['Msun']

    mc = k**(3/5) * (5 / (96 * np.pi**(8/3)))**(3/5) / (G / c**3)
    mc /= msun

    return mc


def calc_k(mc):
    """
    Compute k factor from the chirp mass. The K factor is a 
    stand in for the gravitational wave frequency and its derivative
    with the form k = f**(-11/3)*f'/Msun

    Parameters
    ----------
    mc : float or ndarray
        Chirp mass in solar masses.

    Returns
    -------
    k : float or ndarray
        k parameter(s)
    """
    if not isinstance(mc, (float, np.ndarray)):
        raise TypeError("Input must be float or ndarray")

    if np.any(mc <= 0) or np.any(np.isnan(mc)):
        raise ValueError("Mass must be a positive, non-zero value.")

    cc = constants()

    c = cc['c']
    G = cc['G']
    msun = cc['Msun']

    k = 96/5*np.pi**(8/3)*(G*mc*msun/c**3)**(5/3)
    return k


def calc_fdot_chirp(mc, fgw):
    """
    Spin-up of a chirping gravitational-wave signal (leading PN order).

    Parameters
    ----------
    mc : float
        Chirp mass in solar masses.
    fgw : float
        Gravitational-wave frequency in Hz.

    Returns
    -------
    fdot : float
        Frequency derivative (Hz/s).
    """

    if not isinstance(mc, float):
        raise TypeError("mc must be a float")
    if not isinstance(fgw, float):
        raise TypeError("fgw must be a float")

    if mc <= 0 or np.isnan(mc):
        raise ValueError("mc must be a positive, non-zero value.")
    if fgw <= 0 or np.isnan(fgw):
        raise ValueError("fgw must be a positive, non-zero value.")

    consts = constants()
    G = consts['G']
    c = consts['c']
    Msun = consts['Msun']

    mc = np.asarray(mc, dtype=float)
    fgw = np.asarray(fgw, dtype=float)

    # convert chirp mass to kg
    mc_si = mc * Msun

    fdot = (
        (96.0 / 5.0)
        * np.pi**(8.0 / 3.0)
        * (G * mc_si / c**3)**(5.0 / 3.0)
        * fgw**(11.0 / 3.0)
    )

    return fdot


def calc_time_to_coalescence(Mc, fgw):
    """
    Leading-order time to coalescence for a binary inspiral.

    Parameters
    ----------
    Mc : float
        Chirp mass in solar masses.
    fgw : float
        Gravitational-wave frequency in Hz.

    Returns
    -------
    tau : float
        Time to coalescence (seconds).
    """

    if not isinstance(Mc, float):
        raise TypeError("Mc must be a float")
    if not isinstance(fgw, float):
        raise TypeError("fgw must be a float")

    if Mc <= 0 or np.isnan(Mc):
        raise ValueError("Mc must be a positive, non-zero value.")
    if fgw <= 0 or np.isnan(fgw):
        raise ValueError("fgw must be a positive, non-zero value.")

    consts = constants()
    G = consts['G']
    c = consts['c']
    Msun = consts['Msun']

    Mc = np.asarray(Mc, dtype=float)
    fgw = np.asarray(fgw, dtype=float)

    # convert chirp mass to SI (kg)
    Mc_si = Mc * Msun

    tau = (
        (5.0 / 256.0)
        * (np.pi * fgw)**(-8.0 / 3.0)
        * (G * Mc_si / c**3)**(-5.0 / 3.0)
    )

    return tau


def shift_x0_by_time(x0, kn, delta_t, n):
    """
    Shift x0 by a time offset delta_t.
    Parameters
    ----------
    x0 : float or ndarray
        Initial x0 value(s)
    kn : float or ndarray
        k parameter(s)
    delta_t : float
        Time shift in seconds
    n : float
        Braking index

    Returns
    -------
    x : same type/shape as x0
        Shifted x0
    """

    if not isinstance(x0, (float, np.ndarray)):
        raise TypeError("x0 must be a float or ndarray")

    if not isinstance(kn, (float, np.ndarray)):
        raise TypeError("kn must be a float or ndarray")

    if not isinstance(delta_t, float):
        raise TypeError("delta_t must be a float")

    if not isinstance(n, float):
        raise TypeError("n must be a float")

    if np.any(np.isnan(x0)):
        raise ValueError("All X0 values must be numerical values.")

    if np.any(kn <= 0) or np.any(np.isnan(kn)):
        raise ValueError("All kn values must be greater than zero.")

    # This ensures that the output type condition holds
    if np.size(x0) < np.size(kn):
        raise ValueError("x0 and kn must have the same size")

    if delta_t == 0 or np.isnan(delta_t):
        raise ValueError("Delta_t must no be a non-zero value")

    if np.isnan(n):
        raise ValueError("N must be a numerical value")

    # to decrease frequency (shift from ref time = 0.5 to 0), make delta_t negative so x > x0 meaning f < f0;
    # to increase frequency (shift from ref time = 0 to 0.5), delta_t must be positive so x < x0 --> f > f0
    x0_t = x0 - (n - 1) * np.abs(kn) * delta_t
    if isinstance(x0, float) and isinstance(x0_t, np.ndarray):
        return x0_t[0]
    return x0_t


def get_f0_from_x0(x0, n):
    """
    Compute f0 from x0 given braking index n.

    Parameters
    ----------
    x0 : float or ndarray
        x0 parameter(s)
    n : float
        Braking index

    Returns
    -------
    f0 : same shape as x0
        Recovered frequency
    """

    if not isinstance(x0, (float, np.ndarray)):
        raise TypeError("x0 must be a float or ndarray")

    if not isinstance(n, float):
        raise TypeError("n must be a float")

    if np.any(np.isnan(x0)):
        raise ValueError("All X0 values must be numerical values.")

    if n == 1.0:
        raise ValueError("n MUST NOT be 1.0")

    if np.isnan(n):
        raise ValueError("N must be a numerical value")

    return np.power(x0, -1.0 / (n - 1.0))


def get_x0_from_f0(f0, n):
    """
    Compute x0 from f0 given braking index n.

    Parameters
    ----------
    f0 : float or ndarray
        Frequency value(s)
    n : float
        Braking index

    Returns
    -------
    x0 : same shape as f0
        x0 parameter(s)
    """

    if not isinstance(f0, (float, np.ndarray)):
        raise TypeError("f0 must be a float or ndarray")

    if not isinstance(n, float):
        raise TypeError("n must be a float")

    if np.any(f0 <= 0) or np.any(np.isnan(f0)):
        raise ValueError("Frequency must be a positive, non- zero value")

    if np.isnan(n):
        raise ValueError("N must be a numerical value")

    return 1.0 / np.power(f0, n - 1.0)

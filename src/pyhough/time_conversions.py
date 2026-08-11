import numpy as np

from datetime import datetime, timezone

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
SECONDS_IN_DAY = 86400.0
MJD_AT_GPS_EPOCH = 44244.0 # 6-Jan-1980 00:00:00
TAI_UTC_AT_GPS_EPOCH = 19.0
BASE_TAI_UTC = 10

def gps2mjd(tgps):
    """
    Convert GPS time (seconds) to Modified Julian Date (days).

    Parameters
    ----------
    tgps : float or array-like
        GPS time in seconds.

    Returns
    -------
    mjd : ndarray
        Modified Julian Date (days).
    """
    
    # Reject wrong input types
    if not isinstance(tgps, (float, list, tuple, np.ndarray)):
        raise TypeError

    tgps = np.asarray(tgps, dtype=float)

    mjd = tgps / SECONDS_IN_DAY + MJD_AT_GPS_EPOCH

    # Leap second correction (GPS linked to TAI, offset from UTC)
    mjd = mjd - (leap_seconds(mjd) - TAI_UTC_AT_GPS_EPOCH) / SECONDS_IN_DAY
    
    # Ensure output type consistency
    if not isinstance(mjd, np.ndarray):
        return np.asarray(mjd)

    return mjd


def tdt2tdb(mjd):
    """
    Seconds to add to tdt (terrestrial dymamical time (TAI corrected)) 
    to have tdb (barycentric dynamical time)
    
    Parameters
    ----------
    mjd : int, float, or np.ndarray
        Modified Julian Date (days).

    Returns
    -------
    tdb : float or ndarray
        Seconds to add to the tdt
    """
    if not isinstance(mjd, (int, float, np.ndarray)) or type(mjd) is bool:
        raise TypeError

    JD = mjd + 2400000.5
    g = np.mod(357.53 + 0.98560028 * (JD - 2451545.0),360) * np.pi/180
    tdb = 0.001658 * np.sin(g) + 0.000014 * np.sin(2*g)
    return tdb

def gmst(t):
    """
    Greenwich Mean Sidereal Time (GMST) in hours.

    Parameters
    ----------
    t : float or ndarray
        Time in JD or MJD.
        If t < 1e6, it is assumed to be MJD and converted to JD.

    Returns
    -------
    st : float or ndarray
        GMST in hours (range [0, 24)).
    """

    if not isinstance(t, (float, np.ndarray)) or type(t) is bool:
        raise ValueError

    t = np.asarray(t, dtype=float)

    # Convert MJD -> JD if needed
    jd = np.where(t < 1_000_000, t + 2400000.5, t)

    jd0 = np.floor(jd - 0.5) + 0.5
    h = (jd - jd0) * 24.0

    d  = jd  - 2451545.0
    d0 = jd0 - 2451545.0
    T  = d / 36525.0

    st = (
        6.697374558
        + 0.06570982441908 * d0
        + 1.00273790935 * h
        + 0.000026 * T**2
    )

    return np.mod(st, 24.0)

# print(gmst(6.008867471064815e+04))

def mjuliandate(*args):
    """
    Compute Modified Julian Date.

    Supports:
        mjuliandate(year, month, day)
        mjuliandate(year, month, day, hour, minute, second)
        mjuliandate([[year, month, day], ...])
        mjuliandate([[year, month, day, hour, minute, second], ...])

    Notes
    -----
    Gregorian calendar only. Leap seconds are not included.
    """

    if len(args) == 1:
        arr = np.asarray(args[0], dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] == 3:
            year, month, day = arr[:, 0], arr[:, 1], arr[:, 2]
            hour = np.zeros_like(year)
            minute = np.zeros_like(year)
            second = np.zeros_like(year)

        elif arr.shape[1] == 6:
            year, month, day = arr[:, 0], arr[:, 1], arr[:, 2]
            hour, minute, second = arr[:, 3], arr[:, 4], arr[:, 5]

        else:
            raise ValueError("Single-array input must have shape Nx3 or Nx6.")

    elif len(args) == 3:
        year, month, day = np.broadcast_arrays(
            *[np.asarray([a], dtype=float) for a in args])
        hour = np.zeros_like(year)
        minute = np.zeros_like(year)
        second = np.zeros_like(year)

    elif len(args) == 6:
        year, month, day, hour, minute, second = np.broadcast_arrays(
            *[np.asarray([a], dtype=float) for a in args]
        )

    else:
        raise TypeError("Use mjuliandate(Y,M,D), mjuliandate(Y,M,D,h,m,s), or an Nx3/Nx6 array.")

    year = year.astype(int).copy()
    month = month.astype(int).copy()
    day = day.astype(int).copy()
    hour = hour.astype(int)
    minute = minute.astype(int)
    second = second.astype(float)

    leap_year = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    if np.any(year < 1):
        raise ValueError("This function is intended for CE Gregorian dates only.")
    if np.any(month < 1) or np.any(month > 12):
        raise ValueError("Invalid month value")
    if np.any(day < 1):
        raise ValueError("Invalid day value")
    # Validate specific day number for each month/year combination
    max_day = days_in_month[month - 1]
    max_day[(month == 2) & leap_year] = 29
    if np.any(day > max_day):
        raise ValueError("Invalid day value")
    if np.any(hour < 0) or np.any(hour > 23):
        raise ValueError("Invalid hour value")
    if np.any(minute < 0) or np.any(minute > 59):
        raise ValueError("Invalid minute value")
    if np.any(second < 0) or np.any(second >= 60):
        raise ValueError("Invalid second value")

    jan_feb = month <= 2
    year[jan_feb] -= 1
    month[jan_feb] += 12

    day_fraction = (hour + minute / 60.0 + second / 3600.0) / 24.0

    mjd_day = (
        np.floor(365.25 * year)
        + np.floor(30.6001 * (month + 1.0))
        + 2.0
        - np.floor(year / 100.0)
        + np.floor(np.floor(year / 100.0) / 4.0)
        + day
        - 679006.0
    )

    mjd = mjd_day + day_fraction

    if mjd.size == 1:
        return float(mjd.item())

    return mjd


def utc2gps(*args):
    """
    Convert UTC to GPS seconds.

    Supports:
        utc2gps("2023-05-24T22:00:00")
        utc2gps(year, month, day)
        utc2gps(year, month, day, hour, minute, second)
    """

    if len(args) == 1: 
        if isinstance(args[0], str):
            dt = datetime.fromisoformat(args[0])

            return mjd2gps(
                mjuliandate(
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second + dt.microsecond / 1e6,
                )
            )
        else:
            raise TypeError("utc2gps expects either an ISO string")

    if len(args) == 3:
        year, month, day = args
        hour, minute, second = 0, 0, 0

    elif len(args) == 6:
        year, month, day, hour, minute, second = args

    else:
        raise TypeError(
            "utc2gps expects either "
            "utc2gps(year, month, day), or "
            "utc2gps(year, month, day, hour, minute, second)."
        )

    for arg, name in zip([year, month, day, hour, minute, second], ["year", "month", "day", "hour", "minute", "second"]):
        if not isinstance(arg, (int, float)) or type(arg) is bool:
            raise TypeError(f"{name} must be a numeric value")

    return mjd2gps(
        mjuliandate(year, month, day, hour, minute, second)
    )

def mjd2gps(mjd):
    """
    Convert Modified Julian Date to GPS seconds.

    Parameters
    ----------
    mjd : float or ndarray

    Returns
    -------
    gps : float or ndarray
    """

    return (mjd - MJD_AT_GPS_EPOCH) * SECONDS_IN_DAY + (leap_seconds(mjd) - TAI_UTC_AT_GPS_EPOCH)

def leap_seconds(mjd):
    """
    Return TAI-UTC leap seconds at the given Modified Julian Date.

    Parameters
    ----------
    mjd : float or array-like

    Returns
    -------
    int, float, or ndarray
        TAI-UTC in seconds.
    """

    if not isinstance(mjd,(float, list, tuple, np.ndarray)):
        raise TypeError

    leaptimes = np.array([
        41317,  # 1972 Jan 1,  TAI-UTC = 10
        41499,  # 1972 Jul 1,  TAI-UTC = 11
        41683,  # 1973 Jan 1,  TAI-UTC = 12
        42048,  # 1974 Jan 1,  TAI-UTC = 13
        42413,  # 1975 Jan 1,  TAI-UTC = 14
        42778,  # 1976 Jan 1,  TAI-UTC = 15
        43144,  # 1977 Jan 1,  TAI-UTC = 16
        43509,  # 1978 Jan 1,  TAI-UTC = 17
        43874,  # 1979 Jan 1,  TAI-UTC = 18
        44239,  # 1980 Jan 1,  TAI-UTC = 19
        44786,  # 1981 Jul 1,  TAI-UTC = 20
        45151,  # 1982 Jul 1,  TAI-UTC = 21
        45516,  # 1983 Jul 1,  TAI-UTC = 22
        46247,  # 1985 Jul 1,  TAI-UTC = 23
        47161,  # 1988 Jan 1,  TAI-UTC = 24
        47892,  # 1990 Jan 1,  TAI-UTC = 25
        48257,  # 1991 Jan 1,  TAI-UTC = 26
        48804,  # 1992 Jul 1,  TAI-UTC = 27
        49169,  # 1993 Jul 1,  TAI-UTC = 28
        49534,  # 1994 Jul 1,  TAI-UTC = 29
        50083,  # 1996 Jan 1,  TAI-UTC = 30
        50630,  # 1997 Jul 1,  TAI-UTC = 31
        51179,  # 1999 Jan 1,  TAI-UTC = 32
        53736,  # 2006 Jan 1,  TAI-UTC = 33
        54832,  # 2009 Jan 1,  TAI-UTC = 34
        56109,  # 2012 Jul 1,  TAI-UTC = 35
        57204,  # 2015 Jul 1,  TAI-UTC = 36
        57754,  # 2017 Jan 1,  TAI-UTC = 37
    ], dtype=float)

    tai_minus_utc_values = np.arange(BASE_TAI_UTC, BASE_TAI_UTC + len(leaptimes), dtype=float)

    mjd_arr = np.asarray(mjd, dtype=float)

    indices = np.searchsorted(leaptimes, mjd_arr, side="right") - 1

    if np.any(indices < 0):
        raise ValueError("MJD is before 1972-01-01; leap-second table not defined.")

    nls = tai_minus_utc_values[indices]

    if np.ndim(mjd) == 0:
        return float(nls)

    return nls
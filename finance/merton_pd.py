import numpy as np
from scipy.stats import norm

def merton_pd(V0, D, r, sigma_V, T):
    """
    Modelo Merton:
    V0 = valor de la empresa (market cap)
    D  = deuda total (USD)
    r  = tasa libre riesgo
    sigma_V = volatilidad anual
    T  = años

    PD = N(-d2)
    """
    if V0 <= 0 or D <= 0 or sigma_V <= 0 or T <= 0:
        return None

    d1 = (np.log(V0 / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    pd = norm.cdf(-d2)

    return float(pd), float(d1), float(d2)

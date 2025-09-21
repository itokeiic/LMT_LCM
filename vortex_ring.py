"""
This code reproduces the induced velocity of a vortex ring as described in 
NACA Report 1184: "The Normal Component of the Induced Velocity in the Vicinity 
of a Lifting Rotor" by Heyson and Katzoff.

The implementation follows the mathematical formulation presented in equations 
(1) through (16) of the report.
"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
Gamma = -1.0  # Vortex ring strength (circulation)
R = 1.0      # Vortex ring radius
# --- Core Physics and Math Functions ---
def calculate_wake_strength(CT, omega_R, lambda_val, mu):
    """
    Calculate dΓ/dZ from rotor operating conditions
    
    Parameters:
    -----------
    CT : float
        Thrust coefficient
    omega_R : float  
        Tip speed (Ω·R)
    lambda_val : float
        Inflow ratio
    mu : float
        Advance ratio
    """
    dGamma_dZ = (omega_R * CT) / (lambda_val * (1 - 1.5 * mu**2))
    return dGamma_dZ

def get_elliptic_integrals(k):
    """Wrapper for scipy's elliptic integral functions."""
    # This is a helper function for the elliptic integrals K(k) and E(k)
    # used in Equation (1) and subsequent velocity calculations.
    if k >= 1.0: return np.inf, 1.0
    m = k**2
    return ellipk(m), ellipe(m)

def vz_axis(z, R=1.0, Gamma=1.0):
    """
    Calculates the axial velocity component vz on the axis of a single
    horizontal vortex ring, located at r=0 in the (r,z) plane.

    This is a direct implementation of Equation (16) from NACA TR 1184.
    """
    return [-(0.5 * Gamma / R) * (1 + z**2)**-1.5, 0.0]

def induced_velocity(r, z, R=1.0, Gamma=1.0):
    """
    Calculates the local axial (vz) and radial (vr) velocity component for a single
    HORIZONTAL vortex ring.

    Args:
        r (float): Non-dimensional radial distance from the ring's central 
                   axis, normalized by the ring's radius (r/R).
        z (float): Non-dimensional axial distance from the plane containing 
                   the ring, normalized by the ring's radius (z/R).
        R (float): The radius of the vortex ring.
        Gamma (float): The circulation strength of the vortex ring.

    """
    # Handle on-axis case, which is a direct implementation of Equation (16).
    if np.isclose(r, 0.0):
        # Equation (16): (v_z)_{x=0} = -(Gamma/2R) * (1 / (1 + z^2)^(3/2))
        return vz_axis(z, R, Gamma)

    
    # --- Direct implementation of NACA TR 1184 Equations (5-15) ---
    # Non-dimensional coordinates from report (x=r, z=z)
    x = r

    # Distances d1 and d2 (Eq. 14, 15)
    d1 = max(np.sqrt(z**2 + (x - 1)**2), 1.e-16)  # Avoid division by zero
    d2 = max(np.sqrt(z**2 + (x + 1)**2), 1.e-16)  # Avoid division by zero

    # Modulus tau for elliptic integrals (Eq. 2)
    tau_val = (d2 - d1) / (d2 + d1)
    K_tau, E_tau = get_elliptic_integrals(tau_val)

    # Helper functions A, B, C, D, F (Eq. 7-11)
    A = K_tau - E_tau                                           # Eq (7)
    B = (x - 1) / d1 + (x + 1) / d2                             # Eq (8)
    C = d1 + d2                                                 # Eq (9)
    D = tau_val*E_tau / (1 - tau_val**2)                                # Eq (10)
    
    term_F_1 = (1 + x**2 + z**2) - d1 * d2
    term_F_2 = (1 + x) * d1**2 - (1 - x) * d2**2
    F = 1 - (term_F_1 / (2 * x**2)) - (term_F_2 / (2 * x * d1 * d2)) # Eq (11)
    B_prime = z * (1/d1 + 1/d2)                             # Eq (12)
    F_prime = (z/x)*(1 - (1 + x**2 + z**2)/(d1 * d2))               # Eq (13)
    # Final velocity calculation (Eq. 5)
    # Note: The report's pre-factor is Gamma/(2*pi*R). The 1/x term comes
    # from the derivative of the stream function.
    vz = -(Gamma / (2 * np.pi * x * R)) * (A * B + C * D * F)
    vr = (Gamma / (2 * np.pi * x * R)) * (A * B_prime + C * D * F_prime)
    return [vz, vr]

if __name__ == "__main__":
    # Simple test cases
    print("On-axis (r=0, z=0) (should be 0.5):", induced_velocity(0, 0,R=R,Gamma=Gamma)[0])  
    print("Off-axis (r=1, z=0) (should be inf):", induced_velocity(1, 0,R=R,Gamma=Gamma)[0]) 
    print("Off-axis (r=2, z=0 (should be -0.0431):", induced_velocity(2, 0,R=R,Gamma=Gamma)[0]) 
    print("Off-axis (r=1.4, z=1) (should be 0.0226):", induced_velocity(1.4, 1,R=R,Gamma=Gamma)[0]) 
    print("Off-axis (r=5.0, z=4.2) (should be 0.0002):", induced_velocity(5, 4.2,R=R,Gamma=Gamma)[0])
    print("Off-axis (r=0.7, z=0.0) (should be 0.08461):", induced_velocity(0.7, 0.0,R=R,Gamma=Gamma)[0])


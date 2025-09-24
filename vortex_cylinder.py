import vortex_ring
import numpy as np

# Induced velocity from Total Wake ≈ Σ[i=0→12] (velocity from ring i) × (Simpson weight i) × (dΓ/dZ) × (grid spacing)

def flexible_wake_integration(point_P, R, dGamma_dZ=-1.0,z_max=10.0, n_points=101):
    """
    Flexible wake integration using your vortex ring function
    
    Parameters:
    -----------
    point_P : [x, y, z]
        Point where you want velocity (nondimensionalized by R)
    dGamma_dZ : float
        Wake strength per unit length
    z_max : float
        Non-dimensional maximum wake distance (you choose!)
    n_points : int
        Number of integration points (you choose!)
    """
    
    # Your own grid - uniform spacing for simplicity
    z_wake_positions = np.linspace(0, z_max, n_points)
    delta_z = z_wake_positions[1] - z_wake_positions[0]
    
    # Standard Simpson's rule coefficients
    simpson_coeffs = np.ones(n_points)
    simpson_coeffs[1:-1:2] = 4  # Odd indices
    simpson_coeffs[2:-1:2] = 2  # Even indices
    # simpson_coeffs = [1, 4, 2, 4, 2, ..., 4, 1]
    
    total_velocity = 0
    
    print(f"Integration from z=0 to z={z_max} with {n_points} points")
    print(f"Grid spacing: Δz = {delta_z:.3f}")
    
    for i, z_wake in enumerate(z_wake_positions):
        # Relative position from point P to wake ring
        z_rel = z_wake - point_P[2]
        r_rel = np.sqrt(point_P[0]**2 + point_P[1]**2)
        
        # Your vortex ring function
        
        vz_single, _ = vortex_ring.induced_velocity(r_rel, z_rel, R=R, Gamma=dGamma_dZ)
        # print(f"Debug: R= {R}, dGamma_dZ= {dGamma_dZ}, r_rel= {r_rel}, z_rel= {z_rel:.3f}, vz_single= {vz_single:.6f}")
        # Standard Simpson's rule integration
        contribution = vz_single * simpson_coeffs[i] * delta_z / 3 # vz_single already includes Gamma so dGamma_dZ is not needed here.
        total_velocity += contribution
    
    return total_velocity

def adaptive_wake_integration(point_P, R, dGamma_dZ, tolerance=1e-6, base_n_points=101):
    """
    Keep extending until convergence with adaptive grid density
    
    Parameters:
    -----------
    point_P : [x, y, z]
        Point where you want velocity (nondimensionalized by R)
    R : float
        Vortex ring radius
    dGamma_dZ : float
        Wake strength per unit length
    tolerance : float
        Convergence tolerance
    base_n_points : int
        Base number of points for z_max = 1.0
    """
    z_max = 1.0
    prev_result = 0
    
    while True:
        # Adapt n_points to maintain consistent grid density
        # Scale n_points proportionally with z_max to keep delta_z approximately constant
        n_points = max(int(base_n_points * z_max), base_n_points)
        # Ensure odd number for Simpson's rule
        if n_points % 2 == 0:
            n_points += 1
            
        result = flexible_wake_integration(point_P, R, dGamma_dZ, z_max, n_points)
        
        print(f"z_max = {z_max:.2f}, n_points = {n_points}, result = {result:.6f}")
        
        if abs(result - prev_result) < tolerance:
            print(f"Converged at z_max = {z_max:.2f} with {n_points} points")
            return result
            
        prev_result = result
        z_max *= 1.5  # Extend range
        
        if z_max > 200:  # Safety limit
            print("Reached maximum z_max limit of 100")
            break
    
    return result

if __name__ == "__main__":
    # Example usage
    point_P = [0.7, 0.0, 0.0]  # Point where you want velocity
    R = 1.0                  # Vortex ring radius
    dGamma_dZ = 1           # Wake strength per unit length. Use negative if Z axis is positive upward.
    # dGamma_dZ = -1           # Wake strength per unit length --- IGNORE ---
    
    vz_total = flexible_wake_integration(point_P, R, dGamma_dZ, z_max=4.2, n_points=201)
    print(f"Total induced vz at point P: {vz_total:.6f}")
    vz_total = flexible_wake_integration(point_P, R, dGamma_dZ, z_max=100.0, n_points=2401)
    print(f"Total induced vz at point P (extended): {vz_total:.6f}")
    # vz_total_converged = adaptive_wake_integration(point_P, R, dGamma_dZ, tolerance=1e-7)
    # print(f"Converged induced vz at point P: {vz_total_converged:.6f}")
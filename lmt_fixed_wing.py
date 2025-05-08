import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def elliptical_lift_distribution(L, b, eta):
    """Calculates the lift distribution of an elliptical wing.
    
    Corresponds to Equation (3).
    """
    return (4 * L / (np.pi * b)) * np.sqrt(1 - eta**2)

def f_function(xi):
    """Calculates the f function based on the value of xi."""
    if abs(xi) <= 1:
        return 1
    else:
        return 1 - abs(xi) / np.sqrt(xi**2 - 1)
def f_function_ave(xi_lb,xi_ub):
    integral, _ = quad(f_function,xi_lb,xi_ub)
    return integral/(xi_ub - xi_lb)

def calculate_mi_bar(eta_j, eta_j_plus_1, b_i, b, eta_0i, rho, V):
    """Calculates the effective air mass term mi_bar.
    
    Corresponds to Equation (21).
    """

    def integrand(eta):
        value_inside_sqrt = 1 - (b / b_i)**2 * (eta - eta_0i)**2
        value_inside_sqrt = max(0, value_inside_sqrt)  
        return rho * b_i * V * np.sqrt(value_inside_sqrt)

    # integral, _ = quad(integrand, -eta_j, -eta_j_plus_1)
    integral, _ = quad(integrand, eta_j, eta_j_plus_1)
    # return integral / (eta_j - eta_j_plus_1)
    return integral / (eta_j_plus_1 - eta_j)

def solve_induced_velocities(chord_func, angle_of_attack_func, V, b, n, 
                            placement='symmetric'):
    """
    Solves for the induced velocities using the local momentum theory.
    
    The core logic corresponds to solving the system of equations represented by 
    Equation (22), derived from Equation (20) and using elements from (21).

    Args:
        chord_func: Function that gives chord length as a function of eta.
        angle_of_attack_func: Function for angle of attack as a function of eta.
        V: Forward flight speed.
        b: Wingspan.
        n: Number of elliptical wings. Also, number of sections
        placement: Whether the elliptical wings are symmetrically placed.
                   options are 'symmetric', 'right'.

    Returns:
        A tuple containing:
        - induced_velocities: Array of induced velocities for each section.
        - eta_positions: Spanwise locations of each section.
    """

    A = np.zeros((n, n))
    B = np.zeros(n)
    if placement == 'symmetric':
        eta_positions = np.linspace(-1, 0, n + 1)
        #eta_positions = -np.cos(np.linspace(0.0,np.pi/2,n+1))
    elif placement == 'right':
        eta_positions = np.linspace(-1, 1, n + 1)
        #eta_positions = -1+2*np.sin(np.linspace(0.0,np.pi/2,n+1))
    
    for j in range(n): # for each section
        eta_j = eta_positions[j] #Section lower bound
        eta_j_plus_1 = eta_positions[j + 1] #Section upper bound
        
        eta_pj = (eta_j + eta_j_plus_1) / 2  # Midpoint
        c_j = chord_func(eta_pj)
        theta_j = angle_of_attack_func(eta_pj)

        for i in range(n): # for each elliptic wing
            
            if placement == 'symmetric':
                eta_0i = 0.0 # Symmetric placement. y_0i = 0, so eta_0i = 0
            elif placement == 'right':
                eta_0i = 0.5*(eta_positions[i] + 1.0)

            bi = b*(eta_0i - eta_positions[i])

            if i <= j:
                mi_bar = calculate_mi_bar(eta_j, eta_j_plus_1, bi, b, eta_0i, rho, V)
                A[j, i] = 2 * mi_bar
            else:
                A[j, i] = 0

            def integrand2(eta):
                xi = (eta - eta_0i)*(b/bi)
                return 0.5 * rho * V**2 * c_j * a / V * f_function(xi)
            
            integral2, _ = quad(integrand2, eta_j, eta_j_plus_1)
            A[j,i] = A[j,i] + integral2/(eta_j_plus_1 - eta_j)

        B[j] = 0.5 * rho * V**2 * c_j * a * theta_j  #simplified, induced vel term removed

    dv = np.linalg.solve(A, B) # Solve Equation (22)
    
    #Obtain induced velocities for each section
    induced_velocities = dv.copy() #initializing induced velocity vector
    for j in range(n): # for section j
        eta_j = eta_positions[j] #Section lower bound
        eta_j_plus_1 = eta_positions[j + 1] #Section upper bound
        val = 0.0
        for i in range(n): # for elliptic wing i
            if placement == 'symmetric':
                eta_0i = 0.0 # Symmetric placement. y_0i = 0, so eta_0i = 0
            elif placement == 'right':
                eta_0i = 0.5*(eta_positions[i] + 1.0)
            bi = b*(eta_0i - eta_positions[i])
            xi_lb = (eta_j - eta_0i)*(b/bi)
            xi_ub = (eta_j_plus_1 - eta_0i)*(b/bi)

            val += dv[i]*f_function_ave(xi_lb,xi_ub)
        induced_velocities[j] = val


    return dv, induced_velocities, eta_positions


if __name__ == "__main__":
    # --- Example Usage ---
    rho = 1.225  # Air density
    V = 10       # Flight speed
    #b = 10       # Wingspan
    a = 2 * np.pi # Lift slope
    AR = 10 # Aspect ratio
    S = 10 # Wing area
    taper_ratio = 0.4 # used only in tapered wing
    planform = 'elliptic' # options: 'rectangular', 'elliptic', 'tapered'
    n = 200  # Number of sections
    placement = 'right' # options: 'symmetric', 'right'

    # Example wing geometry (constant chord and angle of attack)
    def constant_aoa(eta):
        return 0.1

    if planform == 'rectangular':
        def constant_chord_function(AR,S):
            chord = S/AR
            b=S/chord
            def chord_func(eta):
                return chord
            return chord_func, b
        chord_func, b = constant_chord_function(AR,S)
    elif planform == 'elliptic':
        def elliptical_chord_function(AR, S):
            b = np.sqrt(AR * S)
            c0 = (4 * S) / (np.pi * b)
            def chord_func(eta):
                # Ensure value inside sqrt is non-negative
                sqrt_val = max(0.0, 1.0 - eta**2) 
                return c0 * np.sqrt(sqrt_val)
            return chord_func, b 
        chord_func, b = elliptical_chord_function(AR,S)
    elif planform == 'tapered':
        def tapered_chord_function(taper_ratio, AR, S):
            b = np.sqrt(AR * S)
            c_root = (2 * S) / (b * (1 + taper_ratio))
            # c_tip = taper_ratio * c_root # Not directly needed in formula below
            def chord_func(eta):
                return c_root * (1.0 - (1.0 - taper_ratio) * abs(eta))
            return chord_func, b
        chord_func, b = tapered_chord_function(taper_ratio,AR,S)

    dv, induced_velocities, eta_positions = solve_induced_velocities(
        chord_func, constant_aoa, V, b, n, placement=placement
    )

    # Calculate lift distribution
    lift_distribution = np.zeros(n)
    induced_drag_distribution = np.zeros(n)
    chord_values = np.zeros(n)
    aoa_values = np.zeros(n)
    total_lift = 0.0
    total_induced_drag = 0.0
    for i in range(n):
        eta_val = (eta_positions[i] + eta_positions[i+1])/2
        chord_values[i] = chord_func(eta_val)
        aoa_values[i] = constant_aoa(eta_val)
        #Uses Equation (17)
        lift_distribution[i] = 0.5 * rho * V**2 * chord_values[i] * a * (aoa_values[i] - induced_velocities[i]/V)
        induced_drag_distribution[i] = lift_distribution[i]*np.sin(np.arctan(induced_velocities[i]/V))
        total_lift += lift_distribution[i]*(eta_positions[i+1]-eta_positions[i])*0.5*b
        total_induced_drag += induced_drag_distribution[i]*(eta_positions[i+1]-eta_positions[i])*0.5*b
    if placement == 'symmetric':
        total_lift = 2*total_lift
        total_induced_drag = 2*total_induced_drag
    CL = total_lift/(0.5*rho*V**2*S)
    CDi = total_induced_drag/(0.5*rho*V**2*S)
    # Print results
    print("dv:",dv)
    print("Induced Velocities:", induced_velocities)
    print("Lift Distribution:", lift_distribution)
    print("Total Lift:",total_lift)
    print("Total Induced Drag:",total_induced_drag)
    print("CL:",CL)
    print("CDi:",CDi)
    print("Span:",b)
    print("Root Chord:",chord_func(0))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot induced velocities
    plt.subplot(1, 2, 1)
    plt.plot(eta_positions[:-1], induced_velocities, label='Induced Velocities', color='orange')
    plt.plot(eta_positions[:-1], [total_lift/(2*rho*V*np.pi*(0.5*b)**2)]*n, label='Induced Velocities, Elliptic Wing')
    plt.xlabel('Spanwise Position (η)')
    plt.ylabel('Induced Velocity (m/s)')
    plt.title('Induced Velocities')
    plt.grid(True)
    plt.legend()

    # Plot lift distribution
    plt.subplot(1, 2, 2)
    
    plt.plot(eta_positions[:-1], lift_distribution, label='Lift Distribution', color='orange')
    plt.plot(eta_positions[:-1], elliptical_lift_distribution(total_lift,b,eta_positions[:-1]),label='Elliptic Lift Distribution')
    plt.xlabel('Spanwise Position (η)')
    plt.ylabel('Lift Distribution (N/m)')
    plt.title('Lift Distribution')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
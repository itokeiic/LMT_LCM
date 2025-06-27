import numpy as np
from scipy.integrate import quad

def calculate_mi_bar(eta_j, eta_j_plus_1, b_i, b, eta_0i, rho, V):
    """Calculates the effective air mass term mi_bar.
    
    Corresponds to Equation (21).
    """

    def integrand(eta):
        value_inside_sqrt = 1 - (b / b_i)**2 * (eta - eta_0i)**2
        value_inside_sqrt = max(0, value_inside_sqrt)  
        return rho * b_i * V * np.sqrt(value_inside_sqrt)

    integral, _ = quad(integrand, eta_j, eta_j_plus_1)
    return integral / (eta_j_plus_1 - eta_j)

def f_function(xi):
    """Calculates the f function based on the value of xi."""
    if abs(xi) <= 1:
        return 1
    else:
        return 1 - abs(xi) / np.sqrt(xi**2 - 1)
def f_function_ave(xi_lb,xi_ub):
    integral, _ = quad(f_function,xi_lb,xi_ub)
    return integral/(xi_ub - xi_lb)

def solve_induced_velocities_rectangular_wing(rho, V, b, c, a, theta):
    """
    Calculates the two induced velocities for the rectangular wing example.

    Args:
        rho: Air density.
        V: Forward flight speed.
        b: Wingspan of the main wing.
        c: Chord length (constant).
        a: Lift slope of the airfoil.
        theta: Geometric angle of attack (constant).

    Returns:
        A tuple containing:
        - dv1: Induced velocity for section 1.
        - dv2: Induced velocity for section 2.
    """
    theta1 = 0
    theta2 = np.pi/3
    b1 = b*np.cos(theta1)
    b2 = b*np.cos(theta2)
    eta_01 = 0
    eta_02 = 0

    # --- Calculate m̄i for both equations ---
    m1_section1 = calculate_mi_bar(-1, -0.5, b1, b, eta_01, rho, V)
    m1_section2 = calculate_mi_bar(-0.5, 0, b1, b, eta_01, rho, V)

    # --- Calculate m̄2 for the second equation ---
    m2_section2 = calculate_mi_bar(-0.5, 0, b2, b, eta_02, rho, V)

    # --- Set up and solve the matrix equation ---
    A = np.array([[2 * m1_section1, 0],
                  [2 * m1_section2, 2 * m2_section2]])
    
    def integrand1(eta):
        xi1 = eta
        xi2 = 2 * eta
        return 0.5 * rho * V**2 * c * a * theta
    
    integral1, _ = quad(integrand1, -1, -0.5)
    B = np.array([integral1, integral1])/(-0.5-(-1))
    
    def integrand2(eta):
        xi1 = eta
        xi2 = 2 * eta
        return 0.5 * rho * V**2 * c * a * theta
    
    integral2, _ = quad(integrand2, -0.5, 0)
    B[1] = integral2/(0-(-0.5))
    
    def integrand3(eta):
        xi1 = eta
        xi2 = 2 * eta
        return 0.5 * rho * V**2 * c * a / V * f_function(xi1)
    
    integral3, _ = quad(integrand3, -1, -0.5)
    A[0,0] = A[0,0] + integral3/(-0.5-(-1))
    
    def integrand4(eta):
        xi1 = eta
        xi2 = 2 * eta
        return 0.5 * rho * V**2 * c * a / V * f_function(xi2)
    
    integral4, _ = quad(integrand4, -1, -0.5)
    A[0,1] = A[0,1] + integral4/(-0.5-(-1))
    
    def integrand5(eta):
        xi1 = eta
        xi2 = 2 * eta
        return 0.5 * rho * V**2 * c * a / V * f_function(xi1)
    
    integral5, _ = quad(integrand5, -0.5, 0)
    A[1,0] = A[1,0] + integral5/(0-(-0.5))
    
    def integrand6(eta):
        xi1 = eta
        xi2 = 2 * eta
        return 0.5 * rho * V**2 * c * a / V * f_function(xi2)
    
    integral6, _ = quad(integrand6, -0.5, 0)
    A[1,1] = A[1,1] + integral6/(0-(-0.5))

    dv = np.linalg.solve(A, B)
    dv1 = dv[0]
    dv2 = dv[1]
    v_ind_section_1 = dv1*f_function_ave(-1,-0.5) + dv2*f_function_ave(-2,-1)
    v_ind_section_2 = dv1*f_function_ave(-0.5,0) + dv2*f_function_ave(-1,0)
    
    l1 = (integral1 - integral3*dv1 - integral4*dv2)/(-0.5-(-1))
    l2 = (integral2 - integral5*dv1 - integral6*dv2)/(0-(-0.5))

    d1 = l1*np.sin(np.arctan(v_ind_section_1/V))
    d2 = l2*np.sin(np.arctan(v_ind_section_2/V))

    return dv1, dv2, l1, l2, d1, d2

# --- Example Usage ---
rho = 1.225  # Air density
V = 10       # Flight speed
b = 10       # Wingspan
c = 1       # Chord length
S=b*c       # Wing area
a = 2 * np.pi # Lift slope
theta = 0.1  # Angle of attack

dv1, dv2, l1, l2, d1, d2= solve_induced_velocities_rectangular_wing(rho, V, b, c, a, theta)
L = 2*(l1*(-0.5-(-1))*0.5*b + l2*(0-(-0.5))*0.5*b)
CL = L/(0.5*rho*V**2*S)

Di = 2*(d1*(-0.5-(-1))*0.5*b + d2*(0-(-0.5))*0.5*b)
CDi = Di/(0.5*rho*V**2*S)

print("Induced Velocity (Section 1):", dv1)
print("Induced Velocity (Section 2):", dv2)
print("Average lift per unit span (Section 1):", l1)
print("Average lift per unit span (Section 2):", l2)
print("Total lift:",L)
print("Lift Coefficient CL:",CL)
print("Total induced drag:",Di)
print("Induced Drag Coefficient CDi:",CDi)

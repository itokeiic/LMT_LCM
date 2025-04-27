import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

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

def solve_induced_velocities_rectangular_wing_n4(rho, V, b, c, a, theta):
    """
    Calculates the induced velocities and lift distribution for the rectangular wing example with n=4.

    Args:
        rho: Air density.
        V: Forward flight speed.
        b: Wingspan of the main wing.
        c: Chord length (constant).
        a: Lift slope of the airfoil.
        theta: Geometric angle of attack (constant).

    Returns:
        A tuple containing:
        - eta_sections: Spanwise locations of the sections.
        - dv: Array of induced velocities for each section.
        - lift_distribution: Array of lift distribution values.
    """

    # Define the spans of the four virtual elliptical wings
    b1 = b
    b2 = 3 * b / 4
    b3 = b / 2
    b4 = b / 4
    eta_01 = 0
    eta_02 = 0
    eta_03 = 0
    eta_04 = 0

    # --- Calculate m̄i for the four sections ---
    m1 = calculate_mi_bar(-1, -0.75, b1, b, eta_01, rho, V)
    m2 = calculate_mi_bar(-0.75, -0.5, b1, b, eta_01, rho, V)
    m3 = calculate_mi_bar(-0.5, -0.25, b1, b, eta_01, rho, V)
    m4 = calculate_mi_bar(-0.25, 0, b1, b, eta_01, rho, V)

    m5 = calculate_mi_bar(-0.75, -0.5, b2, b, eta_02, rho, V)
    m6 = calculate_mi_bar(-0.5, -0.25, b2, b, eta_02, rho, V)
    m7 = calculate_mi_bar(-0.25, 0, b2, b, eta_02, rho, V)

    m8 = calculate_mi_bar(-0.5, -0.25, b3, b, eta_03, rho, V)
    m9 = calculate_mi_bar(-0.25, 0, b3, b, eta_03, rho, V)

    m10 = calculate_mi_bar(-0.25, 0, b4, b, eta_04, rho, V)

    # --- Set up and solve the matrix equation ---
    A = np.array([
        [2 * m1, 0, 0, 0],
        [2 * m2, 2 * m5, 0, 0],
        [2 * m3, 2 * m6, 2 * m8, 0],
        [2 * m4, 2 * m7, 2 * m9, 2 * m10]
    ])

    def integrand1(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a * theta

    integral1, _ = quad(integrand1, -1, -0.75)
    B = np.array([integral1, integral1, integral1, integral1])/(-0.75-(-1))

    def integrand2(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a * theta
    integral2, _ = quad(integrand2, -0.75, -0.5)
    B[1] = integral2/(-0.5-(-0.75))

    def integrand3(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a * theta
    integral3, _ = quad(integrand3, -0.5, -0.25)
    B[2] = integral3/(-0.25-(-0.5))

    def integrand4(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a * theta
    integral4, _ = quad(integrand4, -0.25, 0)
    B[3] = integral4/(0-(-0.25))
    
    def integrand5(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi1)

    integral5, _ = quad(integrand5, -1, -0.75)
    A[0, 0] = A[0, 0] + integral5/(-0.75-(-1))
    
    def integrand6(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi2)
    integral6, _ = quad(integrand6, -1, -0.75)
    A[0, 1] = A[0, 1] + integral6/(-0.75-(-1))
    
    def integrand7(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi3)

    integral7, _ = quad(integrand7, -1.0, -0.75)
    A[0, 2] = A[0, 2] + integral7/(-0.75-(-1))
    
    def integrand8(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi4)
    integral8, _ = quad(integrand8, -1.0, -0.75)
    A[0, 3] = A[0, 3] + integral8/(-0.75-(-1))
    
    def integrand9(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi1)
    integral9, _ = quad(integrand9, -0.75, -0.5)
    A[1, 0] = A[1, 0] + integral9/(-0.5-(-0.75))
    
    def integrand10(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi2)
    integral10, _ = quad(integrand10, -0.75, -0.5)
    A[1, 1] = A[1, 1] + integral10/(-0.5-(-0.75))
    
    def integrand11(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi3)
    integral11, _ = quad(integrand11, -0.75, -0.5)
    A[1, 2] = A[1, 2] + integral11/(-0.5-(-0.75))
    
    def integrand12(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi4)
    integral12, _ = quad(integrand12, -0.75, -0.5)
    A[1, 3] = A[1, 3] - integral12/(-0.5-(-0.75))
    
    def integrand13(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi1)
    integral13, _ = quad(integrand13, -0.5, -0.25)
    A[2, 0] = A[2, 0] + integral13/(-0.25-(-0.5))
    
    def integrand14(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi2)
    integral14, _ = quad(integrand14, -0.5, -0.25)
    A[2, 1] = A[2, 1] + integral14/(-0.25-(-0.5))
    
    def integrand15(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi3)
    integral15, _ = quad(integrand15, -0.5, -0.25)
    A[2, 2] = A[2, 2] + integral15/(-0.25-(-0.5))
    
    def integrand16(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi4)
    integral16, _ = quad(integrand16, -0.5, -0.25)
    A[2, 3] = A[2, 3] + integral16/(-0.25-(-0.5))
    
    def integrand17(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi1)
    integral17, _ = quad(integrand17, -0.25, 0)
    A[3, 0] = A[3, 0] + integral17/(0-(-0.25))

    def integrand18(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi2)
    integral18, _ = quad(integrand18, -0.25, 0)
    A[3, 1] = A[3, 1] + integral18/(0-(-0.25))

    def integrand19(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi3)
    integral19, _ = quad(integrand19, -0.25, 0)
    A[3, 2] = A[3, 2] + integral19/(0-(-0.25))

    def integrand20(eta):
        xi1 = eta
        xi2 = (eta * b) / (3 * b / 4)  # eta * 4/3
        xi3 = (eta * b) / (b / 2)    # eta * 2
        xi4 = (eta * b) / (b / 4)    # eta * 4
        return 0.5 * rho * V**2 * c * a / V * f_function(xi4)
    integral20, _ = quad(integrand20, -0.25, 0)
    A[3, 3] = A[3, 3] + integral20/(0-(-0.25))

    dv = np.linalg.solve(A, B)
    dv1 = dv[0]
    dv2 = dv[1]
    dv3 = dv[2]
    dv4 = dv[3]

    eta_sections = np.array([-0.875, -0.625, -0.375, -0.125])  # Spanwise locations for plotting -0.5 + -0.25/2

    lift_distribution = np.zeros(4)  # Initialize lift distribution array
    lift_distribution[0] = 0.5 * rho * V**2 * c * a * (theta - dv1 / V)
    lift_distribution[1] = 0.5 * rho * V**2 * c * a * (theta - dv2 / V)
    lift_distribution[2] = 0.5 * rho * V**2 * c * a * (theta - dv3 / V)
    lift_distribution[3] = 0.5 * rho * V**2 * c * a * (theta - dv4 / V)

    return eta_sections, dv, lift_distribution

# --- Example Usage ---
rho = 1.225  # Air density
V = 10       # Flight speed
b = 8       # Wingspan
c = 1       # Chord length
a = 2 * np.pi # Lift slope
theta = 0.1  # Angle of attack

eta_sections, dv, lift_distribution = solve_induced_velocities_rectangular_wing_n4(rho, V, b, c, a, theta)

print("Spanwise Locations:", eta_sections)
print("Induced Velocities:", dv)
print("Lift Distribution", lift_distribution)

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Plot 1: Spanwise Lift Distribution
plt.subplot(1, 2, 1)
plt.plot(eta_sections, lift_distribution, marker='o', linestyle='-')
plt.title('Spanwise Lift Distribution')
plt.xlabel('Spanwise Location η')
plt.ylabel('Lift per unit span')
plt.grid(True)

# Plot 2: Spanwise Induced Velocity Distribution
plt.subplot(1, 2, 2)
plt.plot(eta_sections, dv, marker='o', linestyle='-')
plt.title('Spanwise Induced Velocity Distribution')
plt.xlabel('Spanwise Location η')
plt.ylabel('Induced Velocity (m/s)')
plt.grid(True)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline

# --- Constants ---
RHO = 1.225  # Air density (kg/m^3)

class RotorBladeLMT:
    def __init__(self, R, num_blades, chord_dist, twist_dist, pitch_0_75,
                 Omega, num_stations=40, airfoil_lift_slope=2*np.pi):
        self.R = R
        self.b = num_blades # Number of blades
        self.Omega = Omega
        self.num_stations = num_stations

        # Discretize blade
        # x_coords are the START of each segment/root of elliptic wing
        # x_nodes are the midpoints of these segments for BET, and also tip of elliptic wing
        self.x_eta = np.linspace(0.05, 1.0, num_stations + 1) # Nodes from 0.05R to R
        self.x_coords = np.array(self.x_eta[:-1]) # Radial stations (roots of elliptic wings, start of segments)
        self.segment_widths = np.array(np.diff(self.x_eta))
        self.x_mid_points = np.array(self.x_coords + self.segment_widths / 2) # Midpoints of segments
        
        print(f"Debug - x_eta shape: {self.x_eta.shape}")
        print(f"Debug - x_coords shape: {self.x_coords.shape}")
        print(f"Debug - segment_widths shape: {self.segment_widths.shape}")
        print(f"Debug - x_mid_points shape: {self.x_mid_points.shape}")
        print(f"Debug - x_mid_points type: {type(self.x_mid_points)}")
        print(f"Debug - x_mid_points: {self.x_mid_points}")

        if callable(chord_dist):
            self.c_dist = np.array(chord_dist(self.x_mid_points))
        elif isinstance(chord_dist, (int, float)):
            self.c_dist = np.full(self.num_stations, chord_dist)
        else:
            self.c_dist = np.array(chord_dist)

        # Twist distribution and collective pitch
        # twist_dist is function deg(eta), pitch_0_75 is in degrees
        # Total pitch = collective_at_0.75 - twist_at_0.75 + twist_at_eta
        if callable(twist_dist):
            twist_at_0_75 = twist_dist(0.75)
            twist_values = np.array([twist_dist(x) for x in self.x_mid_points])
            self.theta_dist_rad = np.deg2rad(pitch_0_75 - twist_at_0_75 + twist_values)
        else: # Assuming pitch_0_75 is the actual pitch distribution if twist_dist is None
            # Convert pitch_0_75 to radians and create an array of the same size as x_mid_points
            pitch_rad = np.deg2rad(pitch_0_75)
            self.theta_dist_rad = np.full(self.num_stations, pitch_rad, dtype=np.float64)
            print(f"Debug - theta_dist_rad shape: {self.theta_dist_rad.shape}")
            print(f"Debug - theta_dist_rad type: {type(self.theta_dist_rad)}")
            print(f"Debug - theta_dist_rad: {self.theta_dist_rad}")

        if callable(airfoil_lift_slope):
            self.a_lift_slope_dist = airfoil_lift_slope(self.x_mid_points)
        else:
            self.a_lift_slope_dist = np.full_like(self.x_mid_points, airfoil_lift_slope)

        # Elliptic wing properties
        # self.x_coords are the roots x_i of the elliptic wings
        self.b_i = self.R * (1.0 - self.x_coords) # Span of i-th elliptic wing (Eq. 32)

        # Precompute H function integrator components (from Appendix A-1, A-3)
        # g(x,y) = 0.5 * (x*sqrt(1-x^2) + asin(x) - y*sqrt(1-y^2) - asin(y))
        # H(x,y) = -C_i/3 * ((1-x)^1.5 - (1-y)^1.5) + V_ic * g(x,y)
        # For LMT, C_i (related to circulation coefficient) is not used directly like this
        # Instead, H is derived from integrating sqrt(1-xi^2) for m_i (Eq. 47)
        # The H in appendix A.3-7 for m_i comes from int(sqrt(1-xi^2))dx
        # Integral of sqrt(1-x^2)dx = 0.5 * (x*sqrt(1-x^2) + asin(x))
        def H_integrand_func(xi_val):
            # xi_val is a scalar or array
            # Handles cases where xi_val is outside [-1, 1] by clipping for sqrt
            # However, xi should always be within [-1, 1] by definition of elliptic wing
            xi_clipped = np.clip(xi_val, -1.0, 1.0)
            return 0.5 * (xi_clipped * np.sqrt(1 - xi_clipped**2) + np.arcsin(xi_clipped))

        self.H_integrand_func = H_integrand_func
        
        # Data for C_lm (attenuation coefficient) from a typical Fig 14 in PDF
        # Z/R values for columns, r/R values for rows
        # This is a crude digitization. A proper source or function would be better.
        self.clm_r_R_pts = np.array([0.2, 0.6, 0.8, 0.9, 0.95, 0.975, 1.0]) # approximating Fig 14 r/R
        self.clm_Z_R_pts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
        # C_lm values, digitized crudely
        clm_data = np.array([
            [1.0, 0.82, 0.70, 0.60, 0.52, 0.45, 0.35, 0.28, 0.18, 0.13], # r/R=0.2
            [1.0, 0.85, 0.75, 0.67, 0.60, 0.54, 0.43, 0.35, 0.23, 0.17], # r/R=0.6
            [1.0, 0.88, 0.79, 0.72, 0.66, 0.60, 0.49, 0.40, 0.27, 0.20], # r/R=0.8
            [1.0, 0.90, 0.83, 0.77, 0.71, 0.66, 0.55, 0.46, 0.31, 0.23], # r/R=0.9
            [1.0, 0.92, 0.86, 0.81, 0.76, 0.71, 0.60, 0.51, 0.35, 0.26], # r/R=0.95
            [1.0, 0.93, 0.88, 0.84, 0.79, 0.75, 0.64, 0.55, 0.38, 0.29], # r/R=0.975
            [1.0, 0.93, 0.88, 0.84, 0.79, 0.75, 0.64, 0.55, 0.38, 0.29]  # r/R=1.0 (approx)
        ])
        self.clm_interpolator = RectBivariateSpline(self.clm_r_R_pts, self.clm_Z_R_pts, clm_data, kx=1, ky=1)


    def get_C_lm(self, r_R_station, Z_R):
        # r_R_station is eta, Z_R is normalized wake distance
        # Clamp Z_R to avoid extrapolation issues with sparse data
        Z_R_clipped = np.clip(Z_R, self.clm_Z_R_pts.min(), self.clm_Z_R_pts.max())
        r_R_clipped = np.clip(r_R_station, self.clm_r_R_pts.min(), self.clm_r_R_pts.max())
        return self.clm_interpolator(r_R_clipped, Z_R_clipped, grid=False)

    def solve_delta_v_one_blade(self, v_total_at_blade_rad):
        """
        Solves for delta_v_i for a single blade, given the total incident induced velocity.
        Uses Appendix A-3 (Eq. A.3-8, A.3-9).
        v_total_at_blade_rad: induced velocity at blade element midpoints from all sources.
        """
        delta_v = np.zeros(self.num_stations)
        
        # For hover, V_N = 0 (no climb/descent), U_i' = Omega * R * x_i'_c (Eq. 40 simplified)
        # x_i'_c are self.x_mid_points
        U_prime = self.Omega * self.R * self.x_mid_points # Airspeed at segment midpoints

        # Iterate for each elliptic wing / segment (i_prime in PDF notation, k here)
        for k_wing in range(self.num_stations): # k_wing is 0 to num_stations-1
            # This k_wing corresponds to the k-th *strip* in Appendix A-3 terms
            # And we are solving for delta_v[k_wing]
            
            # Properties for the k_wing-th strip (blade element theory part)
            x_k_prime_c = self.x_mid_points[k_wing]
            U_k_prime = U_prime[k_wing]
            theta_k_prime = self.theta_dist_rad[k_wing]
            c_k_prime = self.c_dist[k_wing]
            a_k_prime = self.a_lift_slope_dist[k_wing]
            
            # Inflow angle phi_k_prime (Eq. 45)
            # v_total_at_blade_rad[k_wing] includes self-induction from Delta_v_0 to Delta_v_{k-1}
            # and the wake contribution.
            # For solving Delta_v_k, we need the sum of Delta_v_0 to Delta_v_k (approx)
            # This makes it slightly implicit. Let's use current v_total for phi.
            if U_k_prime < 1e-6 : # Avoid division by zero if Omega or R or x_mid is zero
                phi_k_prime = 0
            else:
                phi_k_prime = v_total_at_blade_rad[k_wing] / U_k_prime

            # Numerator term based on Blade Element Theory (LHS of balance, from Eq. 39)
            # L_bet_strip = 0.5 * RHO * U_k_prime**2 * c_k_prime * a_k_prime * (theta_k_prime - phi_k_prime)
            # This term from Appendix A-3 (A.3-8, A.3-9) is (theta * U - V_N - v_inflow_m)
            # where v_inflow_m is sum of previous delta_v contributions.
            # For hover, V_N = 0. v_inflow_m = sum_{j=0}^{k-1} delta_v[j] * H_contrib_j
            
            # Sum of known delta_v contributions to inflow at station k_wing
            # And sum of m_j * delta_v_j for RHS of Eq 39
            sum_LHS_known_dv_terms = 0 # Sum of Δv_j * H(ξ_kj_upper, ξ_kj_lower) for j < k_wing
            sum_RHS_m_dv_terms = 0 # Sum of 2 * m_j * Δv_j for j < k_wing

            for j_wing in range(k_wing): # j_wing from 0 to k_wing-1
                # Elliptic wing j_wing has root x_j = self.x_coords[j_wing]
                # Strip k_wing is between x_eta[k_wing] and x_eta[k_wing+1]
                
                # V_j_c for m_j (Eq. 33 simplified for hover)
                x_j_root = self.x_coords[j_wing]
                V_j_c = self.Omega * self.R * (1 + x_j_root) / 2.0
                
                # Calculate m_j for elliptic wing j_wing evaluated over strip k_wing (Eq. 47)
                # xi_lower/upper for transforming strip k_wing's bounds to elliptic wing j_wing's coords
                # Eq 35: xi = (2x - (1+x_i))/(1-x_i)
                x_strip_start = self.x_eta[k_wing]
                x_strip_end   = self.x_eta[k_wing+1]

                xi_kj_lower = (2*x_strip_start - (1+x_j_root)) / (1-x_j_root) if (1-x_j_root) !=0 else 0
                xi_kj_upper = (2*x_strip_end - (1+x_j_root)) / (1-x_j_root) if (1-x_j_root) !=0 else 0
                
                # Ensure xi within [-1,1] for physical validity of elliptic wing theory
                xi_kj_lower = np.clip(xi_kj_lower, -1.0, 1.0)
                xi_kj_upper = np.clip(xi_kj_upper, -1.0, 1.0)

                # Integral part for m_j (H function from Appendix A.3-7)
                # This represents integral(sqrt(1-xi^2)) d(xi) * (U_k_prime/V_j_c)
                # but Eq 47 has (Omega*R*x + Vsin(psi)) / V_j_c. For hover, (Omega*R*x_k_prime_c)/V_j_c
                # The H in Appendix A-3 (A.3-6, A.3-7) is directly related to m_i
                # m_i = rho * pi * (R*(1-x_i)/2)^2 * V_i,c * H_from_integral_of_sqrt_1_minus_xi_sq
                # Let's use Eq 47 properly for m_j
                term_H = self.H_integrand_func(xi_kj_upper) - self.H_integrand_func(xi_kj_lower)
                
                # Airspeed ratio for Eq 47: (Omega*R*x_k_prime_c / V_j_c)
                airspeed_ratio_for_mj = (self.Omega * self.R * x_k_prime_c) / V_j_c if V_j_c != 0 else 1.0
                
                m_j_over_strip_k = (RHO * np.pi * (self.b_i[j_wing]/2.0)**2 * V_j_c) * \
                                   airspeed_ratio_for_mj * term_H / self.segment_widths[k_wing] # Average m_j over strip
                
                sum_RHS_m_dv_terms += 2 * m_j_over_strip_k * delta_v[j_wing]

                # Contribution of delta_v[j_wing] to induced velocity at x_k_prime_c
                # This is 1 if x_k_prime_c is within span of elliptic wing j_wing, 0 otherwise
                # Span of elliptic wing j_wing is [x_j_root, 1.0]
                if x_k_prime_c >= x_j_root - 1e-6 : # x_k_prime_c is covered by wing j
                    sum_LHS_known_dv_terms += delta_v[j_wing] 

            # Now solve for delta_v[k_wing]
            # L_bet_strip = sum_RHS_m_dv_terms + 2 * m_k_over_strip_k * delta_v[k_wing]
            # And phi_k_prime depends on delta_v[k_wing]
            # L_bet_strip = A * (theta_k_prime - (sum_LHS_known_dv_terms + delta_v[k_wing]) / U_k_prime)
            # A * (theta_k_prime - sum_LHS_known_dv_terms/U_k_prime - delta_v[k_wing]/U_k_prime) = sum_RHS_m_dv_terms + 2*m_k*delta_v[k_wing]
            # A*theta_mod - A*delta_v[k_wing]/U_k_prime = sum_RHS_m_dv_terms + 2*m_k*delta_v[k_wing]
            # A*theta_mod - sum_RHS_m_dv_terms = delta_v[k_wing] * (A/U_k_prime + 2*m_k)
            
            A_const = 0.5 * RHO * U_k_prime**2 * c_k_prime * a_k_prime
            theta_modified = theta_k_prime - sum_LHS_known_dv_terms / U_k_prime if U_k_prime > 1e-6 else theta_k_prime

            # m_k_over_strip_k (for the k_wing-th elliptic wing itself, over strip k_wing)
            x_k_root = self.x_coords[k_wing]
            V_k_c = self.Omega * self.R * (1 + x_k_root) / 2.0
            
            xi_kk_lower = (2*self.x_eta[k_wing] - (1+x_k_root)) / (1-x_k_root) if (1-x_k_root) !=0 else 0
            xi_kk_upper = (2*self.x_eta[k_wing+1] - (1+x_k_root)) / (1-x_k_root) if (1-x_k_root) !=0 else 0
            xi_kk_lower = np.clip(xi_kk_lower, -1.0, 1.0)
            xi_kk_upper = np.clip(xi_kk_upper, -1.0, 1.0)
            term_H_k = self.H_integrand_func(xi_kk_upper) - self.H_integrand_func(xi_kk_lower)
            airspeed_ratio_for_mk = (self.Omega * self.R * x_k_prime_c) / V_k_c if V_k_c !=0 else 1.0

            m_k_over_strip_k = (RHO * np.pi * (self.b_i[k_wing]/2.0)**2 * V_k_c) * \
                               airspeed_ratio_for_mk * term_H_k / self.segment_widths[k_wing]
            
            numerator = A_const * theta_modified - sum_RHS_m_dv_terms
            denominator = (A_const / U_k_prime if U_k_prime > 1e-6 else 0) + 2 * m_k_over_strip_k
            
            if np.abs(denominator) < 1e-9: # Avoid division by zero / instability
                delta_v[k_wing] = 0 
            else:
                delta_v[k_wing] = numerator / denominator
                
        return delta_v

    def calculate_v_induced_self(self, delta_v):
        """ Calculates v_ijk (self-induced by current blade) at midpoints. Eq 37 """
        v_self = np.zeros(self.num_stations)
        for i_station in range(self.num_stations): # For each station on the blade
            x_eval = self.x_mid_points[i_station]
            current_v_sum = 0
            for k_wing in range(self.num_stations): # Sum contributions from all elliptic wings
                # Elliptic wing k_wing has root x_k = self.x_coords[k_wing]
                # It influences x_eval if x_eval is within its span [x_k, 1.0]
                if x_eval >= self.x_coords[k_wing] -1e-6: # x_eval is covered by wing k
                    current_v_sum += delta_v[k_wing]
            v_self[i_station] = current_v_sum
        return v_self

    def calculate_lift_distribution(self, v_total_rad):
        """ Calculates lift distribution l(x) using Eq 38. """
        U = self.Omega * self.R * self.x_mid_points
        phi = np.zeros_like(U)
        # only calculate phi where U is not zero
        non_zero_U_indices = U > 1e-6
        phi[non_zero_U_indices] = v_total_rad[non_zero_U_indices] / U[non_zero_U_indices]
        
        l_dist = 0.5 * RHO * U**2 * self.c_dist * self.a_lift_slope_dist * (self.theta_dist_rad - phi)
        return l_dist

    def hover_iteration(self, C_T_guess=0.005, use_wake_attenuation=True, fixed_Clm=None,
                        max_iter=100, tol=1e-5):
        """ Iteratively solves for induced velocity in hover. """
        # Initial guess for v_0 (average induced velocity) for C_lm calculation
        v0_guess = self.Omega * self.R * np.sqrt(C_T_guess / 2.0)
        
        delta_tb = (2 * np.pi) / (self.b * self.Omega) # Eq 56
        Z_val = v0_guess * delta_tb # Distance wake element travels
        Z_R_val = Z_val / self.R

        # Initial guess for total induced velocity at blade elements (midpoints)
        v_total_iter = np.full(self.num_stations, v0_guess) # Start with uniform inflow

        print(f"Initial v0_guess: {v0_guess:.3f} m/s, Z/R: {Z_R_val:.3f}")

        for iteration in range(max_iter):
            v_wake_contribution = np.zeros(self.num_stations)
            if use_wake_attenuation and self.b > 1:
                if fixed_Clm is not None:
                    C_lm_values = np.full(self.num_stations, fixed_Clm)
                else:
                    C_lm_values = self.get_C_lm(self.x_mid_points, Z_R_val)
                
                # Wake contribution is from (b-1) other blades, each inducing approx v_total_iter/b
                # Attenuated by C_lm
                v_per_blade_previous_rev = v_total_iter / self.b # Approximate
                v_wake_contribution = C_lm_values * (self.b - 1) * v_per_blade_previous_rev
            
            # The delta_v solver needs the total v_induced at the blade to calculate phi
            # This v_total_iter is from the *previous* global iteration
            delta_v = self.solve_delta_v_one_blade(v_total_iter) 
            
            v_self_new = self.calculate_v_induced_self(delta_v)
            v_total_new = v_self_new + v_wake_contribution

            error = np.linalg.norm(v_total_new - v_total_iter) / (np.linalg.norm(v_total_iter) + 1e-9)
            v_total_iter = 0.7 * v_total_new + 0.3 * v_total_iter # Relaxation

            # Update v0_guess and Z_R_val for C_lm if not fixed
            if not fixed_Clm and use_wake_attenuation:
                v0_guess = np.mean(v_total_iter) # Update v0 based on current total inflow
                Z_val = v0_guess * delta_tb
                Z_R_val = Z_val / self.R
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}: error = {error:.2e}, mean v_total = {np.mean(v_total_iter):.3f}, new Z/R = {Z_R_val:.3f}")

            if error < tol:
                print(f"Converged in {iteration+1} iterations.")
                break
        else:
            print("Warning: Hover iteration did not converge.")
            
        l_dist = self.calculate_lift_distribution(v_total_iter)
        
        # Calculate Thrust and CT
        thrust = np.sum(l_dist * self.segment_widths * self.R) * self.b # Total thrust for all blades
        CT = thrust / (RHO * np.pi * self.R**2 * (self.Omega * self.R)**2)
        print(f"Calculated Thrust: {thrust:.2f} N, CT: {CT:.6f}")

        return self.x_mid_points, l_dist, v_total_iter, CT

# --- Example Usage based on Figure 16 parameters ---
# (Rotor A from Table 1, Page 29, used for Fig 28, 29, 30, similar to Fig 16)
# Fig 16: mu=0, b=2, theta_t=0. (This is simpler)
# Parameters for Fig 16 from its caption and general knowledge:
# R_fig16 = 1.045  # m (from Table 1, experiment A, though Fig 16 itself is not in Table 1 directly)
                # The caption for Fig 16 mentions "Experiment (NACA TN 2953)"
                # TN 2953: R = 2.5 ft = 0.762 m. Let's use this.
R_fig16 = 0.762 # m
num_blades_fig16 = 2
# Chord for TN 2953: sigma = 0.0637, c = sigma * pi * R / b
#chord_fig16 = 0.0637 * np.pi * R_fig16 / num_blades_fig16 # Approx 0.076 m
chord_fig16 = 0.0762 # m
Omega_rpm_fig16 = 800 # RPM for some tests in TN2953
Omega_fig16 = Omega_rpm_fig16 * (2 * np.pi / 60) # rad/s

# Twist for Fig 16 is 0_t = 0 deg. Collective pitch not explicitly stated, but results shown for 0_0.75
# We will aim for a CT to match one of the experimental points.
# Let's take pitch_0_75 = 8 degrees (from Figure 34, which uses TN2953 data)
pitch_coll_fig16 = 8.0 # degrees at 0.75R

def zero_twist(eta):
    return 0.0

print(f"Fig 16 Parameters: R={R_fig16:.3f}m, b={num_blades_fig16}, c={chord_fig16:.4f}m, Omega={Omega_fig16:.2f} rad/s")

rotor_fig16 = RotorBladeLMT(R=R_fig16, num_blades=num_blades_fig16,
                            chord_dist=chord_fig16,
                            twist_dist=zero_twist, # Zero twist
                            pitch_0_75=pitch_coll_fig16, # Collective pitch at 0.75R
                            Omega=Omega_fig16,
                            num_stations=100, # As per PDF text for Fig 16
                            airfoil_lift_slope=5.73) # Common value, 2pi is ~6.28

# Case 1: Solid line (non-uniform C_lm, i.e., use_wake_attenuation=True, fixed_Clm=None)
# For this, we need an initial C_T guess. A typical light helicopter CT is around 0.004-0.006
# From TN2953 fig 7a, for 8 deg pitch, CT is around 0.0045
print("\n--- Running for Fig 16 Solid Line (Calculated C_lm) ---")
x_eta1, l_dist1, v_ind1, CT1 = rotor_fig16.hover_iteration(C_T_guess=0.0045, use_wake_attenuation=True, fixed_Clm=None, tol=1e-4)

# Case 2: Dash-dot line (uniform C_lm = C* = 0.80)
print("\n--- Running for Fig 16 Dash-Dot Line (Fixed C_lm = 0.80) ---")
x_eta2, l_dist2, v_ind2, CT2 = rotor_fig16.hover_iteration(C_T_guess=0.0045, use_wake_attenuation=True, fixed_Clm=0.80, tol=1e-4)


# --- Plotting ---
plt.figure(figsize=(10, 7))
# Convert lift per unit span (N/m) to kg/m as in PDF (l_pdf = l_code / 9.81)
# Or, the PDF might be l / (rho * Omega^2 * R^3) which is a non-dimensional lift.
# Let's plot l / (rho * (Omega*R)^2 * R) = l / (rho * Omega^2 * R^3) to match y-axis of Fig 28-37
# Figure 16 has l(kg/m). So l_plot = l_dist / 9.81.
# And x-axis is r/R.

norm_factor_fig16 = 1.0 / 9.81 # To kg/m
# norm_factor_fig16 = 1.0 / (RHO * Omega_fig16**2 * R_fig16**3) # For non-dim like Fig 28

plt.plot(x_eta1, l_dist1 * norm_factor_fig16, 'b-', label=f'LMT Calculated C_lm (CT={CT1:.4f})')
plt.plot(x_eta2, l_dist2 * norm_factor_fig16, 'g-.', label=f'LMT Fixed C_lm=0.80 (CT={CT2:.4f})')

# Add digitized experimental data from Fig 16 for theta_0.75=8 deg (approximate)
# This data is very roughly estimated from the PDF image.
exp_r_R_fig16 = np.array([0.325, 0.460, 0.590, 0.660, 0.725, 0.790, 0.825, 0.860, 0.890, 0.925, 0.960, 0.977])
exp_l_LB_IN_fig18_digitized = np.array([0.05, 0.09, 0.135, 0.165, 0.21, 0.25, 0.275, 0.3, 0.32, 0.355, 0.335, 0.23]) # Digitized LB/IN values
conversion_LB_IN_to_KGF_M = (4.44822 / 0.0254) / 9.81 # Factor = 17.85186
exp_l_kgf_m_fig16 = exp_l_LB_IN_fig18_digitized * conversion_LB_IN_to_KGF_M
print(f"Debug - exp_l_kgf_m_fig16: {exp_l_kgf_m_fig16}")
plt.plot(exp_r_R_fig16, exp_l_kgf_m_fig16, 'ro', markerfacecolor='none', label='Experiment Fig 18 (800 RPM, 8deg coll.)')
plt.xlabel('r/R')
plt.ylabel('Lift Distribution, l (kg-force/m)') # Clarify unitplt.title(r'LMT Hover - Comparison with Fig 16 (NACA TN 2953 data, $\Theta_{0.75} \approx 8^\circ$)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(0, 1)
plt.savefig('lift_distribution.pdf', bbox_inches='tight', dpi=300)

plt.figure(figsize=(10, 5))
plt.plot(x_eta1, v_ind1, 'b-', label=f'Induced Vel. (Calc C_lm, mean={np.mean(v_ind1):.2f} m/s)')
plt.plot(x_eta2, v_ind2, 'g-.', label=f'Induced Vel. (Fixed C_lm=0.80, mean={np.mean(v_ind2):.2f} m/s)')
plt.xlabel('r/R')
plt.ylabel('Induced Velocity (m/s)')
plt.title('Induced Velocity Distribution in Hover')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(0, 1)
plt.savefig('induced_velocity.pdf', bbox_inches='tight', dpi=300)
plt.show()
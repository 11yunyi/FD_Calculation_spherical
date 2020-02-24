import numpy as np
from numpy import linalg as LA
from Classes.Common.Matrix import SparseMatrix
from Classes.Solve.Solve import Solve
from Classes.MaterialData.MaterialData import P91
from Classes.MaterialData.MaterialData import Steam
from Classes.Data.Data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# initialize Data class
modeltransients = Data()


"""
Solve the temperature distribution for a hollow sphere
"""

"""
Units:
length: mm
time: s
weight: t

cp: J/(kg * K) ==> 10**-6 (mm**2)/(s**2 * K) 
k: W/(s**2 * K) ==> (t*mm)/(s**3 * K)
rho: kg/m**3 ==> 10**-12 (t)/(mm**3)

alpha: W/(m**2*K)

[W] = [10**3 t*mm/s]
"""



# initialize the solve class
solution = Solve()
# initialize the P91 class
p91 = P91()
# initialize the steam class
steam = Steam()
# value for the stop criterion for the converged solution
eps = 1E-5
# inner radius [mm]; float
r_i = 220.
# outer radius [mm]; float
r_o = 310.
# spatial discretization
# n_space: int
n_space = 51
# delta_r: float
delta_r = (r_o - r_i)/(n_space - 1)
# time increment
delta_t = 1.
num_timesteps = int(modeltransients.get_lc4_endtime()/delta_t)
# array of times
t_arr = np.linspace(0., modeltransients.get_lc4_endtime(), num_timesteps+1)
# array of positions in x-direction
r_arr = np.linspace(r_i, r_o, n_space)
# matrix of solutions
T_mat = np.zeros((n_space, num_timesteps+1))
# vector of mean temperature
T_m_vec = np.zeros(num_timesteps+1)
# vector of inner temperatures
T_i_vec = np.zeros(num_timesteps+1)
# vectors for the stresses
f_tang_p_in_vec = np.zeros(num_timesteps+1)
f_tang_p_out_vec = np.zeros(num_timesteps+1)
f_tang_t_in_vec = np.zeros(num_timesteps+1)
f_tang_t_out_vec = np.zeros(num_timesteps+1)
f2_vec = np.zeros(num_timesteps+1)


# Data for calculation according to EN 12952-3
# in
alpha_sp_in = 2.9
d_ms_in = 530.
e_ms_in = 90.
alpha_t_in = 0.96
# out
alpha_sp_out = 3.6
d_ms_out = 530.
e_ms_out = 90.
alpha_t_out = 0.91


# loop over all time-steps
for i_time in range(num_timesteps):
    if i_time == 0:
        # apply initial condition from model transients
        init_temperature = modeltransients.get_lc4_data(0.)[3]
        # set vectors
        T_0 = init_temperature*np.ones(n_space)
        T_new = T_0
        # add initial condition to matrix of solutions
        T_mat[:, 0] = T_0
        # add initial temperatures to the vectors of mean and inner temperatures
        T_i_vec[0] = init_temperature
        T_m_vec[0] = init_temperature
        # Calculation according to EN 12952-3
        # pressure in MPa
        pressure = modeltransients.get_lc4_data((i_time) * delta_t)[2] / 10.
        f2_vec[i_time] = pressure
        f_tang_p_in = alpha_sp_in * (d_ms_in / (4 * e_ms_in)) * pressure
        f_tang_p_out = alpha_sp_out * (d_ms_out / (4 * e_ms_out)) * pressure
        # Calculate thermal stresses
        f_tang_t_in = 0
        f_tang_t_out = 0
        # add values to vectors
        f_tang_p_in_vec[i_time] = f_tang_p_in
        f_tang_p_out_vec[i_time] = f_tang_p_out
        f_tang_t_in_vec[i_time] = 0
        f_tang_t_out_vec[i_time] = 0
    # copy the time-vector at the start of the calculation for the time-step
    T_start = T_new
    # boolean variable to indicate continuing to iterate to get an equilibrium
    iterate = True
    # set iteration variable
    num_iterations = 0
    # loop over all equilibrium iterations
    while iterate:
        # increment the number of iterations
        num_iterations += 1
        # calculate the current time
        time = (i_time+1)*delta_t
        # initialize sparse matrix with the dimension (n_space x n_space)
        K = SparseMatrix(n_space)
        # f-vector without boundary conditions is the negative temperature vector at the start
        f = -T_start
        # Picard-Iteration: use the temperatures calculated in the last iteration
        # for the steel data and steam data in this iteration
        T = T_new
        # loop over all the points n_space
        for i_space in range(n_space):
            if i_space == 0:
                # application of the boundary condition for the convective heat transfer (first point)
                # get the relevant temperatures
                # temperature for the current point
                T_cur = T[i_space]
                # get the relevant material data for the steel
                # heat conductance for the current point
                k_cur = p91.tc(T_cur)
                # heat capacity for the current point
                cp = p91.specific_heat(T_cur)
                # density for the current point
                rho_steel = p91.density(T_cur)
                #
                # Data from the model transient
                #
                steam_temperature = modeltransients.get_lc4_data((i_time+1)*delta_t)[3]
                steam_pressure = modeltransients.get_lc4_data((i_time+1)*delta_t)[2]
                # get the relevant material data for the steam
                [rho_steam, nu_steam, Pr_steam, k_steam] = steam.parameter(steam_pressure, steam_temperature)
                # heat flux calculation
                # calculate volume flow [mm*3/s]
                steam_massflow = modeltransients.get_lc4_data((i_time+1)*delta_t)[1]
                vol_flow = steam_massflow / rho_steam
                #print("rho: {}, massflow: {}, volflow: {}".format(rho_steam, steam_massflow, vol_flow))

                # calculation of the heat transfer coefficient in [10**-3 t/(K*s**3)]
                # calculate current velocity
                cur_velocity = vol_flow / (np.pi * r_i ** 2)
                # calculate current reynolds number
                Re = (2 * cur_velocity * r_i) / nu_steam
                # print( "Re: {}, velocity: {}".format(Re, cur_velocity))
                # calculation of the Nusselt number
                Nu = 0.023 * Re ** 0.8 * Pr_steam ** 0.3
                # calculation of the heat transfer coefficient
                alpha = Nu * k_steam / (2 * r_i)
                #print("alpha: {}".format(alpha))
                # calculate the network Biot and Fourier numbers
                Bi_N = alpha*delta_r/k_cur
                #print "Biot-Number: {}".format(Bi)
                Fo_N = k_cur / (cp*rho_steel) * delta_t / delta_r ** 2
                #print "Fourier-Number: {}".format(Fo)
                # entries for the matrix
                K.insert(np.array([[-(1 + 2*Fo_N*(1 + Bi_N)), 2*Fo_N], [i_space, i_space], [i_space, i_space + 1]]))
                f[0] = f[0] - 2*Fo_N*Bi_N*steam_temperature
            elif i_space == n_space-1:
                # application for the boundary condition for the adiabatic wall (last point)
                # get the relevant temperatures
                # temperature for the current point
                T_cur = T[i_space]
                # heat conductance for the current point
                k_cur = p91.tc(T_cur)
                # heat capacity for the current point
                cp = p91.specific_heat(T_cur)
                # density for the current point
                rho_steel = p91.density(T_cur)
                # calculate the network Fourier number
                Fo_N = k_cur / (cp*rho_steel) * delta_t / delta_r ** 2
                K.insert(np.array([[2*Fo_N, -(1+2*Fo_N)], [i_space, i_space], [i_space - 1, i_space]]))
                # adiabatic wall, no addition to the f-vector needed
            else:
                # get the relevant temperatures
                # temperature for the current point
                T_cur = T[i_space]
                # temperature for the next point
                T_fol = T[i_space + 1]
                # temperature for the previous point
                T_prev = T[i_space - 1]
                # get the relevant material data for the steel
                # heat conductance for the current point
                k_cur = p91.tc(T_cur)
                # heat conductance for the following point
                k_fol = p91.tc(T_fol)
                # heat conductance for the previous point
                k_prev = p91.tc(T_prev)
                # get the difference of heat conductance between the current and the following point
                delta_k = k_fol - k_cur
                # density for the current point
                rho_steel = p91.density(T_cur)
                # heat capacity for the current point
                cp = p91.specific_heat(T_cur)
                # current radius
                r_cur = r_i + i_space * delta_r
                # get coefficients for the matrix
                A = delta_t/(delta_r**2) * k_cur/(rho_steel * cp)
                B = delta_t/(r_cur * delta_r) * k_cur/(rho_steel * cp)
                C = delta_t/(4 * delta_r**2) * (k_fol - k_prev)/(rho_steel * cp)
                # add the equations for the points on the inside
                K.insert(np.array([[A - B - C, -(2*A + 1), A + B + C], [i_space, i_space, i_space],
                                   [i_space - 1, i_space, i_space + 1]]))
                # no boundary condition on the inside
                # no items added to the f-vector, it's already taken care of
        # solve the equation system K * T = f for T
        T_new = solution.solve(K.getmatrix(), f)
        #print (T_new)
        #print f
        # check if equilibrium is reached by calculating the 2-norm
        residual = LA.norm(T-T_new)
        print("Step: {} Iteration: {} Residual: {}".format(i_time+1, num_iterations, residual))
        if residual < eps:
            # solution is deemed to have converged, set boolean iteration-variable to False to stop iterating
            iterate = False
            # print for Debug-purposes
            #print ("Step {}: Equilibrium reached after {} iteration".format(i_time, num_iterations))
            # create graph of solution
            #plt.plot(np.linspace(126., 198., n_space), T_new)
            #plt.savefig('result_{}.pdf'.format(i_time+1), dpi=300)
            #raise Exception('exit')
            # add solution to the array of solutions
            T_mat[:, i_time+1] = T_new
            # add inner temperature to the vector of inner temperatures
            T_i = T_new[0]
            T_i_vec[i_time+1] = T_i
            # Calculate mean temperature
            int1 = 0
            int2 = 0
            for j_space in range(n_space-1):
                int1 += delta_r * (r_arr[j_space] ** 2 + r_arr[j_space + 1] ** 2) / 2
                int2 += delta_r * (r_arr[j_space] ** 2 + r_arr[j_space + 1] ** 2) / 2 * (T_new[j_space] + T_new[j_space+1]) / 2
            T_m = int2/int1
            T_m_vec[i_time+1] = T_m
            # Calculation according to EN 12952-3
            # pressure in MPa
            pressure = modeltransients.get_lc3_data((i_time+1)*delta_t)[2]/10.
            f2_vec[i_time+1] = pressure
            f_tang_p_in = alpha_sp_in * (d_ms_in / (4 * e_ms_in)) * pressure
            f_tang_p_out = alpha_sp_out * (d_ms_out / (4 * e_ms_out)) * pressure
            #print alpha_sp_out * (d_ms_out / (4 * e_ms_out))
            # Calculate thermal stresses
            f_tang_t_in = alpha_t_in * ((p91.te(T_m) * p91.elastic_modulus(T_m)) / (1 - p91.poissons_number(T_m))) * (
                        T_m - T_i)
            f_tang_t_out = alpha_t_out * ((p91.te(T_m) * p91.elastic_modulus(T_m)) / (1 - p91.poissons_number(T_m))) * (
                        T_m - T_i)
            # add values to vectors
            f_tang_p_in_vec[i_time+1] = f_tang_p_in
            f_tang_p_out_vec[i_time + 1] = f_tang_p_out
            f_tang_t_in_vec[i_time + 1] = f_tang_t_in
            f_tang_t_out_vec[i_time + 1] = f_tang_t_out
        elif num_iterations == 64:
            # solutions didn't converge in an appropriate amount of iterations --> abort
            print("Step {}: Calculation didn't converge. Stop!".format())
            #raise Exception('exit')
"""
tmat, xmat = np.meshgrid(t_arr, r_arr)
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(xmat, tmat, T_mat)
plt.xlabel('<---R--->')
plt.ylabel('<---T--->')
plt.title('H(X,T)')
plt.savefig('lc1_temperature.png')
plt.show()
"""
"""
plt.plot(t_arr, T_i_vec, 'k-')
plt.plot(t_arr, T_m_vec, 'r-')
plt.savefig('lc2_temperatures.png')
plt.clf()
plt.plot(t_arr, f_tang_p_in_vec)
plt.plot(t_arr, f_tang_p_out_vec)
plt.plot(t_arr, f_tang_t_in_vec)
plt.plot(t_arr, f_tang_t_out_vec)
plt.savefig('lc2_stresses.png')
print("f_tang_p_in max: {}".format(max(f_tang_p_in_vec)))
print ("f_tang_p_in min: {}".format(min(f_tang_p_in_vec)))
print ("f_tang_p_out max: {}".format(max(f_tang_p_out_vec)))
print ("f_tang_p_out min: {}".format(min(f_tang_p_out_vec)))
print ("f_tang_t_in max: {}".format(max(f_tang_t_in_vec)))
print ("f_tang_t_in min: {}".format(min(f_tang_t_in_vec)))
print ("f_tang_t_out max: {}".format(max(f_tang_t_out_vec)))
print ("f_tang_t_out min: {}".format(min(f_tang_t_out_vec)))
"""
f_1_vec_in = f_tang_p_in_vec + f_tang_t_in_vec
f_1_vec_out = f_tang_p_out_vec + f_tang_t_out_vec
delta_f_12_vec_in = f_1_vec_in + f2_vec
delta_f_12_vec_out = f_1_vec_out + f2_vec
stress_range_in = max(delta_f_12_vec_in) - min(delta_f_12_vec_in)
stress_range_out = max(delta_f_12_vec_out) - min(delta_f_12_vec_out)
print( stress_range_in, stress_range_out)


#print(f_tang_p_in_vec)


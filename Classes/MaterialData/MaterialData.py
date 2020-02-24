from iapws import IAPWS97
import numpy as np


class MaterialData:
    def __init__(self):
        pass


# Class which contains steam properties. The calculation is done by the iapws package and based of IAPWS-IF97
# and the helmholtz equations therein
class Steam:
    def __init__(self):
        pass

    def parameter(self, pressure, temperature):
        # pressure: Input in bar
        # temperature: Input in degC
        steam = IAPWS97(P=pressure / 10., T=temperature + 273.15)
        # return: density (rho) [t/mm**3], kinematic viscosity (nu) [mm**2/s], Prandtl number (Pr) [-],
        # thermal conductivity (k) [W/(m*K)=(t*mm)/(s**3*K)]
        #print [steam.rho*10**-12, steam.nu*10**6, steam.Pr, steam.k, steam.P, steam.T]
        #raise Exception('exit')
        return [steam.rho*10**-12, steam.nu*10**6, steam.Pr, steam.k]


class P91:
    def __init__(self):
        pass

    # return the elastic modulus in MPa for a given temperature
    def elastic_modulus(self, temperature):
        elastic_moduli = [220000, 213000, 208000, 205000, 201000, 198000, 195000,
                          191000, 187000, 183000, 179000, 174000, 168000]
        temperatures = [-75, 25, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        emod_val = np.interp(temperature, temperatures, elastic_moduli)
        return emod_val

    # return the poission's number for a given temperature
    def poissons_number(self, temperature):
        poissons_numbers = [0.3, 0.3]
        temperatures = [-75, 600]
        poissons_number_val = np.interp(temperature, temperatures, poissons_numbers)
        return poissons_number_val

    # return the thermal conductivity in (t * mm)/(s**3 * K)
    def tc(self, temperature):
        # thermal conductivity data in W/(m*K)
        tcs = [22.3, 23.1, 23.8, 24.4, 25, 25.5, 25.9, 26.3, 26.6, 26.9, 27.2, 27.4, 27.5, 27.7, 27.8, 27.9, 27.9,
               27.9, 27.9, 27.9, 27.9, 27.8, 27.7, 27.6]
        temperatures = [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                        500, 525, 550, 575, 600]
        tc_val = np.interp(temperature, temperatures, tcs)
        # return the value for the thermal conductivity in [(t*mm)/s**3] = [W/(m*K)]
        return tc_val

    # return the thermal expansion coefficient in 1/K
    def te(self, temperature):
        tes = [0.0000105, 0.0000106, 0.0000107, 0.0000109, 0.000011, 0.0000111, 0.0000112, 0.0000113, 0.0000114,
               0.0000115, 0.0000116, 0.0000117, 0.0000118, 0.0000119, 0.0000119, 0.000012, 0.0000121, 0.0000122,
               0.0000123, 0.0000123, 0.0000124, 0.0000125, 0.0000126, 0.0000127]
        temperatures = [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                        500, 525, 550, 575, 600]
        te_val = np.interp(temperature, temperatures, tes)
        # return the value for the thermal expansion coefficient in [1/K]
        return te_val

    # return the thermal diffusivity in mm^2/s
    def td(self, temperature):
        tds = [6.61, 6.67, 6.71, 6.74, 6.76, 6.76, 6.75, 6.71, 6.66, 6.58, 6.49, 6.39, 6.27, 6.15, 6.01, 5.87, 5.72,
               5.56, 5.4, 5.22, 5.04, 4.85, 4.64, 4.42]*10**-6
        temperatures = [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                        500, 525, 550, 575, 600]
        td_val = np.interp(temperature, temperatures, tds)
        # return value for the density in 10**6[mm**2/s] = [m**2/s]
        return td_val*10**6

    # return the density in t/mm**3
    def density(self, temperature):
        # density data in kg/m^3
        densities = [7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750,
                     7750, 7750, 7750, 7750, 7750, 7750, 7750, 7750]
        temperatures = [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                        500, 525, 550, 575, 600]
        density_val = np.interp(temperature, temperatures, densities)
        # return value for the density in 10**-12*[t/mm*3] = [kg/m**3]
        return density_val*10**-12

    # return the specific heat in (mm**2)/(s**2 * K)
    def specific_heat(self, temperature):
        # specific heat data in J/(kg*K)
        specific_heats = [435.3130643, 446.8733375, 457.6703043, 467.1197473, 477.1903035, 486.7341096, 495.1015532,
                          505.7449161, 515.3540637, 527.5026963, 540.782345, 553.2838609, 565.9309564, 581.1696827,
                          596.8547045, 613.2879046, 629.3706294, 647.4820144, 666.6666667, 689.6551724, 714.2857143,
                          739.6075823, 770.3003337, 805.7217924]
        temperatures = [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                        500, 525, 550, 575, 600]
        specific_heat_val = np.interp(temperature, temperatures, specific_heats)
        # return value for the specific heat in 10**-6*[mm**2/(K*s)] = [mm**2/(K*s)]
        return specific_heat_val*10**6

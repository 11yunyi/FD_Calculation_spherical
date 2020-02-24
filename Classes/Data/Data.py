import numpy as np
from Classes.Common.CSV import _CSV
import numpy as np
import os

class Data:

    def __init__(self):
        self.f_steam = ""

    def coldstart(self):
        time = np.array([0, 5100, 10200, 11400, 13800, 15000, 30600, 32700, 35400])
        load = np.array([0, 0, 0, 320, 350, 450, 450, 750, 800])
        steam_massflow = np.array([0, 200, 200, 280, 320, 380, 380, 620, 670])
        steam_pressure = np.array([0, 100, 100, 100, 120, 140, 140, 230, 250])
        steam_temperature = np.array([110, 420, 420, 420, 430, 460, 540, 540, 540])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def warmstart(self):
        time = np.array([0, 5700, 6300, 7200, 8700, 12300, 14400, 17700, 18300, 19200])
        load = np.array([0, 0, 250, 250, 450, 450, 600, 600, 750, 800])
        steam_massflow = np.array([0, 240, 240, 240, 370, 370, 490, 490, 640, 650])
        steam_pressure = np.array([0, 100, 100, 100, 140, 140, 190, 190, 240, 250])
        steam_temperature = np.array([250, 425, 425, 425, 460, 510, 520, 530, 530, 530])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def hotstart(self):
        time = np.array([0, 1200, 16200, 16800, 27600, 28800, 34200])
        load = np.array([0, 0, 0, 400, 400, 600, 800])
        steam_massflow = np.array([0, 280, 280, 280, 400, 580, 680])
        steam_pressure = np.array([130, 130, 130, 130, 150, 210, 250])
        steam_temperature = np.array([450, 480, 510, 515, 550, 550, 550])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def loadchange1(self):
        time = np.array([0, 300, 1200, 2100, 2700, 4200])
        load = np.array([320, 345, 560, 575, 705, 800])
        steam_massflow = np.array([320, 340, 490, 500, 620, 680])
        steam_pressure = np.array([125, 130, 185, 190, 230, 250])
        steam_temperature = np.array([520, 520, 525, 525, 540, 550])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def get_lc1_data(self, time_val):
        [time, load, steam_massflow, steam_pressure, steam_temperature] = self.loadchange1()
        load_val = np.interp(time_val, time, load)
        steam_massflow_val = np.interp(time_val, time, steam_massflow)
        steam_pressure_val = np.interp(time_val, time, steam_pressure)
        steam_temperature_val = np.interp(time_val, time, steam_temperature)
        return [load_val, steam_massflow_val, steam_pressure_val, steam_temperature_val]

    def get_lc1_endtime(self):
        return float(self.loadchange1()[0][-1])


    def loadchange2(self):
        time = np.array([0, 600, 1200, 1800])
        load = np.array([640, 740, 750, 800])
        steam_massflow = np.array([520, 640, 640, 680])
        steam_pressure = np.array([200, 240, 240, 250])
        steam_temperature = np.array([545, 550, 550, 550])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def get_lc2_data(self, time_val):
        [time, load, steam_massflow, steam_pressure, steam_temperature] = self.loadchange2()
        load_val = np.interp(time_val, time, load)
        steam_massflow_val = np.interp(time_val, time, steam_massflow)
        steam_pressure_val = np.interp(time_val, time, steam_pressure)
        steam_temperature_val = np.interp(time_val, time, steam_temperature)
        return [load_val, steam_massflow_val, steam_pressure_val, steam_temperature_val]

    def get_lc2_endtime(self):
        return float(self.loadchange2()[0][-1])

    def loadchange3(self):
        time = np.array([0, 2100, 3300])
        load = np.array([800, 370, 330])
        steam_massflow = np.array([670, 320, 300])
        steam_pressure = np.array([250, 125, 120])
        steam_temperature = np.array([550, 530, 530])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def get_lc3_data(self, time_val):
        [time, load, steam_massflow, steam_pressure, steam_temperature] = self.loadchange3()
        load_val = np.interp(time_val, time, load)
        steam_massflow_val = np.interp(time_val, time, steam_massflow)
        steam_pressure_val = np.interp(time_val, time, steam_pressure)
        steam_temperature_val = np.interp(time_val, time, steam_temperature)
        return [load_val, steam_massflow_val, steam_pressure_val, steam_temperature_val]

    def get_lc3_endtime(self):
        return float(self.loadchange3()[0][-1])

    def loadchange4(self):
        time = np.array([0, 900, 1500])
        load = np.array([800, 650, 640])
        steam_massflow = np.array([660, 530, 520])
        steam_pressure = np.array([250, 200, 200])
        steam_temperature = np.array([550, 550, 550])
        return [time, load, steam_massflow, steam_pressure, steam_temperature]

    def get_lc4_data(self, time_val):
        [time, load, steam_massflow, steam_pressure, steam_temperature] = self.loadchange4()
        load_val = np.interp(time_val, time, load)
        steam_massflow_val = np.interp(time_val, time, steam_massflow)
        steam_pressure_val = np.interp(time_val, time, steam_pressure)
        steam_temperature_val = np.interp(time_val, time, steam_temperature)
        return [load_val, steam_massflow_val, steam_pressure_val, steam_temperature_val]

    def get_lc4_endtime(self):
        return float(self.loadchange4()[0][-1])

    def steamdata(self):
        # steam object
        steam = _CSV("{}/Classes/Data/steamdata.csv".format(os.getcwd()))
        # get steam-data
        steam_data = []
        for row in steam.reader():
            steam_data.append(row)
        steam.close()
        steam_data = np.array(steam_data, dtype=np.float)
        time = steam_data[:, 0]
        massflow = steam_data[:, 1]
        pressure = steam_data[:, 2]
        temperature = steam_data[:, 3]
        return [time, massflow, pressure, temperature]
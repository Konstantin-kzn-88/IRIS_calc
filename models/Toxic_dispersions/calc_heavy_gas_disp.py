# -----------------------------------------------------------
# Класс предназначен для расчета выброса тяжелого газа
#
# Fires, explosions, and toxic gas dispersions :
# effects calculation and risk analysis /
# Marc J. Assael, Konstantinos E. Kakosimos.

# (C) 2022 Kuznetsov Konstantin, Kazan , Russian Federation
# email kuznetsovkm@yandex.ru
# -----------------------------------------------------------

import math

GRAVITY = 9.81  # ускорение свободного падения м/с2
KGM3_TO_MGLITER = 1000  # кг/м3 в мг/л


class Instantaneous_source:
    def __init__(self, wind_speed: float, density_air: float, density_init: float, volume_gas: float):
        """
        Класс предназначен для расчета выброса тяжелого газа

        :paramwind_speed - скорость ветра, м/с
        :paramdensity_air - плотность воздуха, кг/м3
        :param density_init: - плотность газа, кг/м3
        :param volume_gas: - объем газа, м3
        """
        self.wind_speed = wind_speed
        self.density_air = density_air
        self.density_init = density_init
        self.volume_gas = volume_gas

    def alpha(self) -> float:
        '''
        Britter and McQuaid diagram - апроксимация

        :return: alpha_aprox - параметр для апроксимации графика (альфа-параметр)
        Figure C5.20. Britter and McQuaid diagram [Britter & McQuaid 1988]
        '''
        g0 = (GRAVITY * (self.density_init - self.density_air)) / self.density_air
        alpha_aprox = (1 / 2) * math.log10(g0 * math.pow(self.volume_gas, 1 / 3) / math.pow(self.wind_speed, 2))
        return alpha_aprox

    def beta(self, alpha_aprox: float, concentration: float) -> float:
        '''
        Britter and McQuaid diagram - апроксимация
        :param alpha_aprox: - параметр для апроксимации графика (альфа-параметр)
        :param concentration: - концентрация, кг/м3
        :return: beta_aprox - параметр для апроксимации графика (бета-параметр)
        Figure C5.20. Britter and McQuaid diagram [Britter & McQuaid 1988]
        '''
        if concentration == 0.1:
            if alpha_aprox <= -0.44:
                beta_aprox = 0.7
            elif alpha_aprox > -0.44 and alpha_aprox <= 0.43:
                beta_aprox = 0.26 * alpha_aprox + 0.81
            elif alpha_aprox > 0.43 and alpha_aprox <= 1:
                beta_aprox = 0.93
            else:
                beta_aprox = 0.93

        elif concentration == 0.05:
            if alpha_aprox <= -0.56:
                beta_aprox = 0.85
            elif alpha_aprox > -0.56 and alpha_aprox <= 0.31:
                beta_aprox = 0.26 * alpha_aprox + 1
            elif alpha_aprox > 0.31 and alpha_aprox <= 1:
                beta_aprox = -0.12 * alpha_aprox + 1.12
            else:
                beta_aprox = -0.12 * alpha_aprox + 1.12

        elif concentration == 0.02:
            if alpha_aprox <= -0.66:
                beta_aprox = 0.95
            elif alpha_aprox > -0.66 and alpha_aprox <= 0.32:
                beta_aprox = 0.36 * alpha_aprox + 1.19
            elif alpha_aprox > 0.32 and alpha_aprox <= 1:
                beta_aprox = -0.26 * alpha_aprox + 1.38
            else:
                beta_aprox = -0.26 * alpha_aprox + 1.38

        elif concentration == 0.01:
            if alpha_aprox <= -0.71:
                beta_aprox = 1.15
            elif alpha_aprox > -0.71 and alpha_aprox <= 0.37:
                beta_aprox = 0.34 * alpha_aprox + 1.39
            elif alpha_aprox > 0.37 and alpha_aprox <= 1:
                beta_aprox = -0.38 * alpha_aprox + 1.66
            else:
                beta_aprox = -0.38 * alpha_aprox + 1.66

        elif concentration == 0.005:
            if alpha_aprox <= -0.52:
                beta_aprox = 1.48
            elif alpha_aprox > -0.52 and alpha_aprox <= 0.24:
                beta_aprox = 0.26 * alpha_aprox + 1.62
            elif alpha_aprox > 0.24 and alpha_aprox <= 1:
                beta_aprox = -0.30 * alpha_aprox + 1.75
            else:
                beta_aprox = -0.30 * alpha_aprox + 1.75

        elif concentration == 0.002:
            if alpha_aprox <= 0.27:
                beta_aprox = 1.83
            elif alpha_aprox > 0.27 and alpha_aprox <= 1:
                beta_aprox = -0.32 * alpha_aprox + 1.92
            else:
                beta_aprox = -0.32 * alpha_aprox + 1.92

        elif concentration == 0.001:
            if alpha_aprox <= -0.1:
                beta_aprox = 2.075
            elif alpha_aprox > 0.1 and alpha_aprox <= 1:
                beta_aprox = -0.27 * alpha_aprox + 2.05
            else:
                beta_aprox = -0.27 * alpha_aprox + 2.05

        else:
            beta_aprox = -0.27 * alpha_aprox + 2.05

        return beta_aprox

    def find_distance(self, beta_aprox: float) -> float:
        '''
        Функция поиска расстояния для облака
        :param beta_aprox: - параметр для апроксимации графика (бета-параметр)
        :return: x_dist - расстояние, м
        '''
        x_dist = math.pow(10, beta_aprox) * math.pow(self.volume_gas, 1 / 3)
        return x_dist

    def find_time(self, x_dist: float):
        '''

        :param x_dist: - расстояние, м
        :return: (time, b_param) - время, с и ширина облака, м
        '''
        diametr_cloud = math.pow(self.volume_gas / math.pi, 1 / 3)
        g0 = (GRAVITY * (self.density_init - self.density_air)) / self.density_air
        time = 0
        while True:
            if math.fabs((math.sqrt(diametr_cloud * diametr_cloud + 1.2 * time * math.sqrt(g0 * self.volume_gas))) - (
                    x_dist - 0.4 * self.wind_speed * time)) < 0.1:
                return (time, x_dist - 0.4 * self.wind_speed * time)
            time += 0.01

    def concentration(self):
        '''

        :param diametr_cloud: - диаметр облака начальный, м
        :return: (data_concentration, data_x_dist, data_width, data_time)
        концентрации кг/м3;
        расстояние, м
        ширина облака, м
        время подхода облака, с

        '''
        data_concentration = []
        data_x_dist = []
        data_width = []
        data_time = []

        limit_concentration = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)

        for item_concentration in limit_concentration:
            alpha = self.alpha()
            beta = self.beta(alpha, item_concentration)
            x_dist = self.find_distance(beta)
            conc_in_dist = self.density_init * item_concentration
            time_arr = self.find_time(x_dist)
            print('item_concentration')
            time_cloud = time_arr[0]
            b_eq = time_arr[1]

            data_concentration.append(conc_in_dist)
            data_x_dist.append(x_dist)
            data_width.append(b_eq)
            data_time.append(time_cloud)

        return (data_concentration, data_x_dist, data_width, data_time)

    def toxic_dose(self, concentration: float):
        '''
        :param concentration - концентрация, кг/м3
        :return: (dose): float: - токсодоза, мг*мин/л (переход взят из методики ТОКСИ2)
        '''
        dose = ((2 * math.pow(2 * math.pi, 2)) / self.wind_speed) * concentration * 16.67
        return dose

    def result(self):
        data = self.concentration()
        return [[self.toxic_dose(i) for i in data[0]], data[0], data[1], data[2], data[3]]


class Continuous_source:
    def __init__(self, wind_speed: int, density_air: float, density_init: float, gas_flow: float, radius_flow: float):
        """
        Класс предназначен для расчета выброса тяжелого газа

        :param  wind_speed - скорость ветра, м/с
        :param density_air - плотность воздуха, кг/м3
        :param density_init: - плотность газа, кг/м3
        :param gas_flow: - расход газа, кг/с
        :param radius_flow: - радиус выброса, м


        """
        self.wind_speed = wind_speed
        self.density_air = density_air
        self.density_init = density_init
        self.gas_flow = gas_flow
        self.radius_flow = radius_flow

    def alpha(self) -> float:
        '''
        Britter and McQuaid diagram - апроксимация
        :return: alpha_aprox - параметр для апроксимации графика (альфа-параметр)
        Figure C5.17. Britter and McQuaid diagram [Britter & McQuaid 1988].
        '''
        volumetric_consumption_gas = self.gas_flow / self.density_init
        g0 = (9.81 * (self.density_init - 1.21)) / 1.21
        alpha_aprox = (1 / 5) * math.log10(g0 * g0 * volumetric_consumption_gas / math.pow(self.wind_speed, 5))
        return alpha_aprox

    def beta(self, alpha_aprox: float, concentration: float) -> float:
        '''
        Britter and McQuaid diagram - апроксимация
        :param alpha_aprox: - параметр для апроксимации графика (альфа-параметр)
        :param concentration: - концентрация, кг/м3
        :return: beta_aprox - параметр для апроксимации графика (бета-параметр)
        Figure C5.17. Britter and McQuaid diagram [Britter & McQuaid 1988].
        '''
        if concentration == 0.1:
            if alpha_aprox <= -0.55:
                beta_aprox = 1.75
            elif alpha_aprox > -0.55 and alpha_aprox <= -0.14:
                beta_aprox = 0.24 * alpha_aprox + 1.88
            elif alpha_aprox > -0.14 and alpha_aprox <= 1:
                beta_aprox = 0.5 * alpha_aprox + 1.78
            else:
                beta_aprox = 0.5 * alpha_aprox + 1.78

        elif concentration == 0.05:
            if alpha_aprox <= -0.68:
                beta_aprox = 1.92
            elif alpha_aprox > -0.68 and alpha_aprox <= -0.29:
                beta_aprox = 0.36 * alpha_aprox + 2.16
            elif alpha_aprox > -0.29 and alpha_aprox <= -0.18:
                beta_aprox = 2.06
            elif alpha_aprox > -0.18 and alpha_aprox <= 1:
                beta_aprox = 0.56 * alpha_aprox + 1.96
            else:
                beta_aprox = 0.56 * alpha_aprox + 1.96

        elif concentration == 0.02:
            if alpha_aprox <= -0.69:
                beta_aprox = 2.08
            elif alpha_aprox > -0.69 and alpha_aprox <= -0.31:
                beta_aprox = 0.45 * alpha_aprox + 2.39
            elif alpha_aprox > -0.31 and alpha_aprox <= -0.16:
                beta_aprox = 2.25
            elif alpha_aprox > -0.16 and alpha_aprox <= 1:
                beta_aprox = 0.54 * alpha_aprox + 2.16
            else:
                beta_aprox = 0.54 * alpha_aprox + 2.16

        elif concentration == 0.01:
            if alpha_aprox <= -0.70:
                beta_aprox = 2.25
            elif alpha_aprox > -0.70 and alpha_aprox <= -0.29:
                beta_aprox = 0.49 * alpha_aprox + 2.59
            elif alpha_aprox > -0.29 and alpha_aprox <= -0.20:
                beta_aprox = 2.45
            elif alpha_aprox > -0.20 and alpha_aprox <= 1:
                beta_aprox = 0.52 * alpha_aprox + 2.35
            else:
                beta_aprox = 0.52 * alpha_aprox + 2.35

        elif concentration == 0.005:
            if alpha_aprox <= -0.67:
                beta_aprox = 2.4
            elif alpha_aprox > -0.67 and alpha_aprox <= -0.28:
                beta_aprox = 0.59 * alpha_aprox + 2.80
            elif alpha_aprox > -0.28 and alpha_aprox <= -0.15:
                beta_aprox = 2.63
            elif alpha_aprox > -0.15 and alpha_aprox <= 1:
                beta_aprox = 0.49 * alpha_aprox + 2.56
            else:
                beta_aprox = 0.49 * alpha_aprox + 2.56

        elif concentration == 0.002:
            if alpha_aprox <= -0.69:
                beta_aprox = 2.6
            elif alpha_aprox > -0.69 and alpha_aprox <= -0.25:
                beta_aprox = 0.39 * alpha_aprox + 2.87
            elif alpha_aprox > -0.25 and alpha_aprox <= -0.13:
                beta_aprox = 2.77
            elif alpha_aprox > -0.13 and alpha_aprox <= 1:
                beta_aprox = 0.50 * alpha_aprox + 2.71
            else:
                beta_aprox = 0.50 * alpha_aprox + 2.71

        else:
            beta_aprox = 0.50 * alpha_aprox + 2.71

        return beta_aprox

    def find_distance(self, beta_aprox: float) -> float:
        '''
        Функция поиска расстояния для облака
        :param beta_aprox: - параметр для апроксимации графика (бета-параметр)
        :return: x_dist - расстояние, м
        '''
        volumetric_consumption_gas = self.gas_flow / self.density_init
        x_dist = math.pow(10, beta_aprox) * math.pow(volumetric_consumption_gas / self.wind_speed, 0.5)
        return x_dist

    def concentration(self):
        '''
        :return: (data_concentration, data_x_dist, data_width, data_time)
        концентрации кг/м3;
        расстояние, м
        ширина облака, м

        '''
        data_concentration = []
        data_x_dist = []
        data_width = []

        volumetric_consumption_gas = self.gas_flow / self.density_init

        limit_concentration = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002)

        for item_concentration in limit_concentration:
            g0 = (GRAVITY * (self.density_init - self.density_air)) / self.density_air
            alpha = self.alpha()
            beta = self.beta(alpha, item_concentration)
            x_dist = self.find_distance(beta)
            conc_in_dist = self.density_init * item_concentration
            l_b = g0 * volumetric_consumption_gas / math.pow(self.wind_speed, 3)
            widht_plume = 2 * self.radius_flow + 8 * l_b + 2.5 * math.pow(l_b, 1 / 3) * math.pow(x_dist, 2 / 3)

            data_concentration.append(conc_in_dist)
            data_x_dist.append(x_dist)
            data_width.append(widht_plume)

        return (data_concentration, data_x_dist, data_width)

    def toxic_dose(self, concentration: float):
        '''
        :param concentration - концентрация, кг/м3
        :return: (dose): float: - токсодоза, мг*мин/л (переход взят из методики ТОКСИ2)
        '''
        dose = ((2 * math.pow(2 * math.pi, 2)) / self.wind_speed) * concentration * 16.67
        return dose

    def result(self):
        data = self.concentration()
        return [[self.toxic_dose(i) for i in data[0]], data[0], data[1], data[2]]


if __name__ == '__main__':
    # cls = Instantaneous_source(wind_speed=4, density_air=1.21, density_init=6, volume_gas=10)
    # print(cls.concentration())

    cls = Continuous_source(wind_speed=1, density_air=1.21, density_init=6, gas_flow=1)
    print(cls.concentration())

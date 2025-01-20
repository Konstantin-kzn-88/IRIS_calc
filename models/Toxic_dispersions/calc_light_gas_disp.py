# -----------------------------------------------------------
# Класс предназначен для расчета выброса легкого газа
#
# Fires, explosions, and toxic gas dispersions :
# effects calculation and risk analysis /
# Marc J. Assael, Konstantinos E. Kakosimos.

# Методика Токси 2

# (C) 2022 Kuznetsov Konstantin, Kazan , Russian Federation
# email kuznetsovkm@yandex.ru
# -----------------------------------------------------------

import math


class Source:
    #
    def __init__(self, ambient_temperature: float, cloud: float,
                 wind_speed: float,
                 is_night: float, is_urban_area: float, ejection_height: float,
                 gas_temperature: float, gas_weight: float, gas_flow: float, closing_time: float,
                 molecular_weight: float):
        """
        :param ambient_temperature - температура окружающего воздуха, град. С
        :param cloud - облачность от 0 до 8, -
        :param wind_speed - скорость ветра, м/с
        :param is_night - ночное время суток, -
        :param is_urban_area - городская застройка, -
        :param ejection_height - высота выброса, м
        :param gas_temperature - температура газа, град.С
        :param gas_weight - масса газа, кг
        :param gas_flow - расход газа, кг/с
        :param closing_time - время отсечения, с
        :param molecular_weight - молекулярная масса, кг/кмоль (метан 18)

        """
        self.ambient_temperature = ambient_temperature
        self.cloud = cloud if cloud in [i for i in range(0, 9)] else 0
        self.wind_speed = wind_speed
        self.is_night = is_night if is_night in (0, 1) else 0
        self.is_urban_area = is_urban_area if is_urban_area in (0, 1) else 0
        self.ejection_height = ejection_height
        self.gas_temperature = gas_temperature
        self.gas_weight = gas_weight
        self.gas_flow = gas_flow
        self.closing_time = closing_time
        self.molecular_weight = molecular_weight

        # вычисленные
        self.gas_density = self.molecular_weight / (22.413 * (1 + 0.00367 * self.gas_temperature))  # НПБ 105-03
        self.pasquill = self.pasquill_atmospheric_stability_classes()
        self.radius_first_cloud = math.pow((3 / (4 * math.pi)) * (self.gas_weight / self.gas_density), 1 / 3)


    def pasquill_atmospheric_stability_classes(self) -> str:
        """
        Классы стабильности атмосферы по Паскуиллу
        Table C5.2. Pasquill Atmospheric Stability Classes. p.222
        :return: : pasquill_class: str: класс атмосферы
        """

        table_data = (
            ('A', 'B', 'B', 'F', 'F'),
            ('B', 'B', 'C', 'E', 'F'),
            ('B', 'B', 'C', 'D', 'E'),
            ('C', 'D', 'D', 'D', 'D'),
            ('C', 'D', 'D', 'D', 'D')
        )

        if self.wind_speed < 2:
            wind_ind = 0
        elif self.wind_speed >= 2 and self.wind_speed < 3:
            wind_ind = 1
        elif self.wind_speed >= 3 and self.wind_speed < 5:
            wind_ind = 2
        elif self.wind_speed >= 5 and self.wind_speed < 6:
            wind_ind = 3
        else:
            wind_ind = 4

        if self.cloud <= 2 and not self.is_night:
            cloud_ind = 0
        elif self.cloud <= 5 and self.cloud > 2 and not self.is_night:
            cloud_ind = 1
        elif self.cloud <= 8 and self.cloud > 5 and not self.is_night:
            cloud_ind = 2
        elif self.cloud <= 3 and self.is_night:
            cloud_ind = 3
        else:
            cloud_ind = 4

        pasquill_class = table_data[wind_ind][cloud_ind]

        return pasquill_class

    def stability_coefficients(self):
        '''
        Функция подбора коэффициентов в зависимости от класса стабильности атмосферы
        :return: кортеж коэффициентов a_1, a_2, b_1, b_2, c_3 (таблица3, методика Токси 2)
        '''

        coefficients = ((0.112, 0.000920, 0.92, 0.718, 0.11),
                        (0.098, 0.00135, 0.889, 0.688, 0.08),
                        (0.0609, 0.00196, 0.895, 0.0684, 0.06))

        if self.pasquill in ('A', 'B'):
            return coefficients[0]
        elif self.pasquill in ('C', 'D'):
            return coefficients[1]
        else:
            return coefficients[2]

    def roughness_coefficients(self):
        '''
        Функция подбора коэффициентов в зависимости от застройки
        :return: кортеж коэффициентов c_1, c_2, d_1, d_2 (таблица4, методика Токси 2)
        '''

        coefficients = ((1.56, 0.000625, 0.048, 0.045),
                        (7.37, 0.000233, -0.0096, 0.6))

        if self.is_urban_area == 0:
            return coefficients[0]
        else:
            return coefficients[1]

    def dispersion_param(self, x_dist: int):
        '''
        Функция параметров дисперсии
        :param x_dist - дистанция м
        :return: sigma: tuple: кортеж параметров (sigma_x, sigma_y, sigma_z)
        '''

        def __sigma_z_chek(g_x, f_z):
            'Проверка параметра sigma_z по условию табл 5 Токси 2'
            result = g_x * f_z
            if self.pasquill in ('A', 'B'):
                return result if result < 640 else 640
            elif self.pasquill in ('C', 'D'):
                return result if result < 400 else 400
            else:
                return result if result < 220 else 220

        a_1, a_2, b_1, b_2, c_3 = self.stability_coefficients()
        c_1, c_2, d_1, d_2 = self.roughness_coefficients()
        sigma_x = (c_3 * x_dist) / math.sqrt(1 + 0.000 * x_dist)

        sigma_y = sigma_x
        #
        # sigma_y = sigma_x if x_dist / self.wind_speed < 600 else (220.2 * 60 + x_dist / self.wind_speed) / (
        #         220.2 * 60 + 600)

        g_x = (a_1 * math.pow(x_dist, b_1)) / (1 + a_2 * math.pow(x_dist, b_2))

        if self.is_urban_area == 0:
            f_z = math.log((c_1 * math.pow(x_dist, d_1)) * (1 + c_2 * math.pow(x_dist, d_2)))
        else:
            f_z = math.log((c_1 * math.pow(x_dist, d_1)) / (1 + c_2 * math.pow(x_dist, d_2)))

        sigma_z = __sigma_z_chek(g_x, f_z)

        return (sigma_x, sigma_y, sigma_z)

    def concentration(self, x_dist: int) -> float:
        '''
        Функция зависимости концентрации от расстояния
        :param x_dist - дистанция м
        :return: (concentration): float: - концентрация, кг/м3

        '''
        sigma_x, sigma_y, sigma_z = self.dispersion_param(x_dist)

        g_n = math.exp(-math.pow(self.ejection_height, 2) / (2 * math.pow(sigma_z, 2))) * 10
        first_cloud_conc = (self.gas_weight / (
                2.67 * math.pi * math.pow(self.radius_first_cloud, 3) + math.pow(2 * math.pi,
                                                                                 3 / 2) * sigma_x * sigma_y * sigma_z)) * g_n

        second_cloud_conc = ((self.closing_time * self.gas_flow) / (
                2.67 * math.pi * math.pow(self.radius_first_cloud, 3) + math.pow(2 * math.pi,
                                                                                 3 / 2) * sigma_x * sigma_y * sigma_z)) * g_n

        return (first_cloud_conc + second_cloud_conc)

    def toxic_dose(self, x_dist: int):
        '''
        Функция зависимости токсодозы от расстояния
        :param x_dist - дистанция м
        :return: (dose): float: - токсодоза, мг*мин/л
        '''
        concentration = self.concentration(x_dist)
        dose = ((2 * math.pow(2 * math.pi, 2)) / self.wind_speed) * concentration
        return dose

    def find_distance_by_toxdose(self, target_dose: float, tolerance: float = 0.01) -> float:
        """
        Находит расстояние, на котором достигается заданная токсодоза
        :param target_dose: целевая токсодоза, мг*мин/л
        :param tolerance: допустимая погрешность
        :return: расстояние в метрах
        """
        left = 1  # минимальное расстояние
        right = 10000  # максимальное расстояние

        while (right - left) > tolerance:
            mid = (left + right) / 2
            current_dose = self.toxic_dose(int(mid))

            if abs(current_dose - target_dose) < tolerance:
                return mid
            elif current_dose > target_dose:
                left = mid
            else:
                right = mid

        return (left + right) / 2

    def get_threshold_distances(self, threshold_dose: float, lethal_dose: float) -> tuple:
        """
        Находит расстояния для пороговой и смертельной токсодоз
        :param threshold_dose: пороговая токсодоза, мг*мин/л
        :param lethal_dose: смертельная токсодоза, мг*мин/л
        :return: кортеж (расстояние для пороговой токсодозы, расстояние для смертельной токсодозы)
        """
        threshold_distance = self.find_distance_by_toxdose(threshold_dose)
        lethal_distance = self.find_distance_by_toxdose(lethal_dose)

        return threshold_distance, lethal_distance

    def result(self):
        """
        Расчет массива значений концентрации и токсодозы
        :return: кортеж списков (расстояния, концентрации, токсодозы)
        """
        dist = []
        conc = []
        dose = []

        for radius in range(1, 10000):
            c = self.concentration(x_dist=radius)
            d = self.toxic_dose(x_dist=radius)
            dist.append(radius)
            conc.append(c)
            dose.append(d)
            if radius < 100: continue
            if d < 0.1: break

        return (dist, conc, dose)


if __name__ == '__main__':
    # Пример использования
    source = Source(ambient_temperature=10, cloud=0,
                    wind_speed=1, is_night=0, is_urban_area=1, ejection_height=2,
                    gas_temperature=20, gas_weight=0, gas_flow=200/3600, closing_time=300,
                    molecular_weight=17)

    # Пример поиска расстояний для токсодоз
    threshold_dist, lethal_dist = source.get_threshold_distances(threshold_dose=10, lethal_dose=65)
    print(f"Расстояние для пороговой токсодозы: {threshold_dist:.1f} м")
    print(f"Расстояние для смертельной токсодозы: {lethal_dist:.1f} м")
import math
import numpy as np
from enum import Enum
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SpaceClass(Enum):
    """Классы загроможденности пространства"""
    ZAGR_1 = 1  # сильно загроможденное пространство
    ZAGR_2 = 2  # средне загроможденное пространство
    ZAGR_3 = 3  # слабо загроможденное пространство
    ZAGR_4 = 4  # свободное пространство


class SensitivityClass(Enum):
    """Классы чувствительности горючих веществ"""
    CLASS_1 = 1  # особо чувствительные вещества
    CLASS_2 = 2  # чувствительные вещества
    CLASS_3 = 3  # средне чувствительные вещества
    CLASS_4 = 4  # слабо чувствительные вещества


# Давления, для которых необходимо определить расстояния (кПа)
PRESSURES_OF_INTEREST = [100, 70, 28, 14, 5, 2]


def calculate_probit(delta_P: float, impulse: float) -> float:
    """
    Расчет пробит-функции для избыточного давления и импульса
    """
    delta_P_pa = delta_P * 1000
    V = (17500 / delta_P_pa) ** 8.4 + (290 / impulse) ** 9.3
    return 5 - 0.26 * math.log(V)


def probability_from_probit(probit: float) -> float:
    """
    Преобразование пробит-значения в вероятность поражения
    """
    return 0.5 * (1 + math.erf((probit - 5) / math.sqrt(2)))


class TVSExplosion:
    def __init__(self, M_g, q_g, sensitivity_class, space_class, c_g=None, c_st=None,
                 beta=1, is_gas=True, is_ground_level=False):
        self.P0 = 101325  # атмосферное давление, Па
        self.C0 = 340  # скорость звука, м/с

        # Сохраняем параметры как атрибуты класса
        self.M_g = M_g
        self.q_g = q_g
        self.sensitivity_class = sensitivity_class
        self.space_class = space_class
        self.c_g = c_g
        self.c_st = c_st
        self.beta = beta
        self.is_gas = is_gas
        self.is_ground_level = is_ground_level

        # Таблица режимов горения
        self.explosion_mode_table = {
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_1): 1,
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_2): 2,
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_3): 3,
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_4): 3,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_1): 2,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_2): 3,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_3): 4,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_4): 4,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_1): 3,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_2): 4,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_3): 5,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_4): 5,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_1): 4,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_2): 5,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_3): 6,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_4): 6,
        }

    def calculate_energy(self, M_g, q_g, c_g=None, c_st=None, beta=1, is_ground_level=False):
        """Расчет эффективного энергозапаса"""
        q_g = q_g * 1000  # перевод в Дж/кг

        if c_g is None or c_st is None:
            E = M_g * q_g * beta
        else:
            if c_g <= c_st:
                E = M_g * q_g * beta
            else:
                E = M_g * q_g * beta * c_st / c_g

        if is_ground_level:
            E *= 2

        return E

    def get_explosion_mode(self, sensitivity_class: SensitivityClass, space_class: SpaceClass):
        """Определение режима взрывного превращения"""
        return self.explosion_mode_table.get((sensitivity_class, space_class), 6)

    def calculate_flame_velocity(self, M_g, mode):
        """Расчет скорости фронта пламени"""
        if mode == 1:
            return 500
        elif mode == 2:
            return 400
        elif mode == 3:
            return 250
        elif mode == 4:
            return 175
        elif mode == 5:
            return 43 * math.pow(M_g, 1 / 6)
        elif mode == 6:
            return 26 * math.pow(M_g, 1 / 6)
        return None

    def calculate_dimensionless_distance(self, r, E):
        """Расчет безразмерного расстояния"""
        return r / math.pow(E / self.P0, 1 / 3)

    def calculate_pressure_deflagration(self, V_g, R_x, sigma):
        """Расчет безразмерного давления при дефлаграции"""
        term1 = math.pow(V_g / self.C0, 2)
        term2 = (sigma - 1) / sigma
        term3 = 0.83 / R_x - 0.14 / math.pow(R_x, 2)
        return term1 * term2 * term3

    def calculate_impulse_deflagration(self, V_g, R_x, sigma):
        """Расчет безразмерного импульса при дефлаграции"""
        term1 = (V_g / self.C0) * (sigma - 1) / sigma
        term2 = 1 - 0.4 * (sigma - 1) * V_g / (sigma * self.C0)
        term3 = 0.06 / R_x + 0.01 / math.pow(R_x, 2) - 0.0025 / math.pow(R_x, 3)
        return term1 * term2 * term3

    def calculate_final_parameters(self, P_x, I_x, E):
        """Расчет размерных величин давления и импульса"""
        P = P_x * self.P0
        I = I_x * math.pow(self.P0, 2 / 3) * math.pow(E, 1 / 3) / self.C0
        return P, I

    def calculate_explosion_parameters_base(self, r, M_g, q_g, sensitivity_class: SensitivityClass,
                                            space_class: SpaceClass, c_g=None, c_st=None, beta=1,
                                            is_gas=True, is_ground_level=False, include_probit=True):
        """Базовый расчет параметров взрыва"""
        explosion_mode = self.get_explosion_mode(sensitivity_class, space_class)
        sigma = 7 if is_gas else 4
        E = self.calculate_energy(M_g, q_g, c_g, c_st, beta, is_ground_level)
        V_g = self.calculate_flame_velocity(M_g, explosion_mode)
        R_x = self.calculate_dimensionless_distance(r, E)

        R_kr = 0.34
        if R_x < R_kr:
            R_x = R_kr

        P_x = self.calculate_pressure_deflagration(V_g, R_x, sigma)
        I_x = self.calculate_impulse_deflagration(V_g, R_x, sigma)
        P, I = self.calculate_final_parameters(P_x, I_x, E)

        results = {
            'режим_взрывного_превращения [-]': explosion_mode,
            'эффективный_энергозапас [Дж]': E,
            'скорость_фронта_пламени [м/с]': V_g,
            'безразмерное_расстояние [-]': R_x,
            'безразмерное_давление [-]': P_x,
            'безразмерный_импульс [-]': I_x,
            'избыточное_давление [Па]': P,
            'импульс_фазы_сжатия [Па·с]': I
        }

        if include_probit:
            delta_P_kpa = P / 1000
            probit_value = calculate_probit(delta_P_kpa, I)
            probability = probability_from_probit(probit_value)
            results.update({
                'пробит_функция [-]': probit_value,
                'вероятность_поражения [-]': probability
            })

        return results

    def find_distance_for_pressure(self, target_pressure_kpa, M_g, q_g, sensitivity_class: SensitivityClass,
                                   space_class: SpaceClass, c_g=None, c_st=None, beta=1,
                                   is_gas=True, is_ground_level=False, tolerance=0.01):
        """Находит расстояние, на котором достигается заданное избыточное давление"""
        target_pressure = target_pressure_kpa * 1000

        r_min = 1.0
        r_max = 3000.0
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            r = (r_min + r_max) / 2
            params = self.calculate_explosion_parameters_base(
                r, M_g, q_g, sensitivity_class, space_class,
                c_g, c_st, beta, is_gas, is_ground_level, include_probit=False
            )
            current_pressure = params['избыточное_давление [Па]']

            relative_error = abs(current_pressure - target_pressure) / target_pressure
            if relative_error < tolerance:
                return r

            if current_pressure > target_pressure:
                r_min = r
            else:
                r_max = r

            if r_max - r_min < 0.01:
                if r_max < 3000.0:
                    r_max *= 2
                    r_min = r_max / 4
                else:
                    return None

            iteration += 1

        return None

    def plot_explosion_parameters(self, r_min=0.1, r_max=200, num_points=1000):
        """Создание графиков параметров взрыва"""
        # Создаем массив расстояний
        r_linear = np.linspace(r_min, r_max, num_points)
        r_log = np.logspace(np.log10(r_min), np.log10(r_max), num_points)

        # Массивы для хранения результатов
        pressures_linear = []
        pressures_log = []
        impulses = []
        probits = []
        probabilities = []

        # Расчет параметров для каждого расстояния
        for r in r_linear:
            results = self.calculate_explosion_parameters_base(r, self.M_g, self.q_g,
                                                               self.sensitivity_class, self.space_class,
                                                               self.c_g, self.c_st, self.beta,
                                                               self.is_gas, self.is_ground_level)
            pressures_linear.append(results['избыточное_давление [Па]'] / 1000)  # Переводим в кПа

        for r in r_log:
            results = self.calculate_explosion_parameters_base(r, self.M_g, self.q_g,
                                                               self.sensitivity_class, self.space_class,
                                                               self.c_g, self.c_st, self.beta,
                                                               self.is_gas, self.is_ground_level)
            pressures_log.append(results['избыточное_давление [Па]'] / 1000)
            impulses.append(results['импульс_фазы_сжатия [Па·с]'])
            probits.append(results['пробит_функция [-]'])
            probabilities.append(results['вероятность_поражения [-]'] * 100)

        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=(15, 20))

        # 1. График зависимости давления от расстояния (линейный масштаб)
        ax1 = fig.add_subplot(321)
        ax1.plot(r_linear, pressures_linear, 'b-', label='Давление')
        # ax1.axhline(y=700, color='r', linestyle='--', label='Макс. давление (700 кПа)')
        ax1.set_xlabel('Расстояние (м)')
        ax1.set_ylabel('Избыточное давление (кПа)')
        ax1.set_title('Зависимость давления от расстояния\n(линейный масштаб)')
        ax1.grid(True)
        ax1.legend()

        # 2. График зависимости давления от расстояния (логарифмический масштаб)
        ax2 = fig.add_subplot(322)
        ax2.semilogx(r_log, pressures_log, 'b-', label='Давление')
        # ax2.axhline(y=700, color='r', linestyle='--', label='Макс. давление (700 кПа)')

        ax2.set_xlabel('Расстояние (м)')
        ax2.set_ylabel('Избыточное давление (кПа)')
        ax2.set_title('Зависимость давления от расстояния\n(логарифмический масштаб)')
        ax2.grid(True)
        ax2.legend()

        # 3. График зависимости импульса от расстояния
        ax3 = fig.add_subplot(323)
        ax3.semilogx(r_log, impulses, 'g-')
        ax3.set_xlabel('Расстояние (м)')
        ax3.set_ylabel('Импульс (Па·с)')
        ax3.set_title('Зависимость импульса от расстояния')
        ax3.grid(True)

        # 4. График зависимости пробит-функции от расстояния
        ax4 = fig.add_subplot(324)
        ax4.semilogx(r_log, probits, 'r-')
        ax4.set_xlabel('Расстояние (м)')
        ax4.set_ylabel('Значение пробит-функции')
        ax4.set_title('Зависимость пробит-функции от расстояния')
        ax4.grid(True)

        # 5. График зависимости вероятности поражения от расстояния
        ax5 = fig.add_subplot(325)
        ax5.semilogx(r_log, probabilities, 'y-')
        ax5.set_xlabel('Расстояние (м)')
        ax5.set_ylabel('Вероятность поражения (%)')
        ax5.set_title('Зависимость вероятности поражения от расстояния')
        ax5.grid(True)

        # 6. График характерных значений давления
        ax6 = fig.add_subplot(326)
        pressures = [100, 70, 28, 14, 5, 2]
        distances = []
        valid_points = []
        valid_pressures = []

        for p in pressures:
            d = self.find_distance_for_pressure(p, self.M_g, self.q_g,
                                                self.sensitivity_class, self.space_class,
                                                self.c_g, self.c_st, self.beta,
                                                self.is_gas, self.is_ground_level)
            if d is not None:
                distances.append(d)
                valid_points.append((d, p))
                valid_pressures.append(p)

        # Построение точек и линий
        if valid_points:
            x_coords, y_coords = zip(*valid_points)
            ax6.plot(x_coords, y_coords, 'b-')
            ax6.scatter(x_coords, y_coords, color='black', zorder=5)

            # Добавление подписей к точкам
            for x, y in zip(x_coords, y_coords):
                ax6.annotate(f'{y} кПа', (x, y), xytext=(5, 5),
                             textcoords='offset points')

        ax6.set_xlabel('Расстояние (м)')
        ax6.set_ylabel('Давление (кПа)')
        ax6.set_title('Характерные значения давления')
        ax6.grid(True)

        # Настройка общего вида графиков
        plt.tight_layout()

        # Сохранение графиков
        plt.savefig('explosion_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Пример использования класса"""
    # Создаем экземпляр класса с параметрами
    tvs = TVSExplosion(
        M_g=100,  # масса горючего вещества, кг
        q_g=46000,  # теплота сгорания, кДж/кг
        sensitivity_class=SensitivityClass.CLASS_2,  # класс чувствительности
        space_class=SpaceClass.ZAGR_1,  # класс загроможденности
        is_gas=True,  # газовая смесь
        is_ground_level=True  # облако на поверхности земли
    )

    # Построение графиков
    tvs.plot_explosion_parameters(r_min=0.1, r_max=200, num_points=1000)

    # Вывод характерных значений
    print("\nХарактерные значения давления и соответствующие расстояния:")
    for pressure in PRESSURES_OF_INTEREST:
        distance = tvs.find_distance_for_pressure(
            pressure, tvs.M_g, tvs.q_g, tvs.sensitivity_class, tvs.space_class,
            tvs.c_g, tvs.c_st, tvs.beta, tvs.is_gas, tvs.is_ground_level
        )
        if distance is not None:
            print(f"Давление {pressure} кПа достигается на расстоянии {distance:.2f} м")
        else:
            print(f"Давление {pressure} кПа не достигается")


if __name__ == "__main__":
    main()
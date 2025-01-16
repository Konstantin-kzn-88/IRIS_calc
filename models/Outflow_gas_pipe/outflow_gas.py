import numpy as np
from scipy.optimize import fsolve
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Константы
P_ATM = 101325  # Атмосферное давление [Па]


class GasLeakModel:
    def __init__(self, gas_properties):
        """
        Инициализация модели истечения газа

        Parameters:
        gas_properties (dict): Свойства газа
            - k: показатель адиабаты
            - R: газовая постоянная [Дж/(кг·К)]
            - T_C: температура газа [°C]
            - M: молярная масса [кг/моль]
        """
        self.k = gas_properties['k']
        self.R = gas_properties['R']
        # Перевод температуры из °C в K
        self.T = gas_properties['T_C'] + 273.15
        self.M = gas_properties['M']

    def critical_pressure_ratio(self):
        """Расчет критического отношения давлений"""
        return (2 / (self.k + 1)) ** (self.k / (self.k - 1))

    def mass_flow_rate(self, P1_MPa, D_mm, discharge_coef=0.61):
        """
        Расчет массового расхода газа через отверстие

        Parameters:
        P1_MPa (float): Давление в трубе [МПа]
        D_mm (float): Диаметр отверстия [мм]
        discharge_coef (float): Коэффициент истечения

        Returns:
        float: Массовый расход [кг/с]
        """
        # Конвертация давления в Па и диаметра в м
        P1 = P1_MPa * 1e6
        D_m = D_mm / 1000.0
        P2 = P_ATM

        A = np.pi * (D_m / 2) ** 2

        pr_crit = self.critical_pressure_ratio()
        pr = P2 / P1

        if pr <= pr_crit:  # Критическое истечение
            pr = pr_crit

        # Формула для расчета массового расхода
        if pr < 1:  # Проверка физического смысла
            term1 = 2 * self.k / (self.k - 1)
            term2 = (pr ** (2 / self.k) - pr ** ((self.k + 1) / self.k))
            G = discharge_coef * A * np.sqrt(self.k * P1 * self.R * self.T * term1 * term2)
            return G
        else:
            return 0

    def velocity(self, P1_MPa):
        """
        Расчет скорости истечения газа

        Parameters:
        P1_MPa (float): Давление в трубе [МПа]

        Returns:
        float: Скорость истечения [м/с]
        """
        # Конвертация давления в Па
        P1 = P1_MPa * 1e6
        P2 = P_ATM

        if P2 >= P1:  # Нет истечения при обратном перепаде давления
            return 0

        pr = P2 / P1
        pr_crit = self.critical_pressure_ratio()

        if pr <= pr_crit:
            # Критический режим - скорость равна местной скорости звука
            v = np.sqrt(self.k * self.R * self.T)
        else:
            # Докритический режим
            v = np.sqrt(2 * self.k / (self.k - 1) * self.R * self.T * (1 - pr ** ((self.k - 1) / self.k)))

        return v

    def plot_pressure_dependence(self, D_mm, P1_range_MPa):
        """
        Построение графиков зависимости расхода и скорости от давления в трубе

        Parameters:
        D_mm (float): Диаметр отверстия [мм]
        P1_range_MPa (array): Массив значений давления в трубе [МПа]
        """
        plt.style.use('default')
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5

        mass_flows = [self.mass_flow_rate(P1, D_mm) for P1 in P1_range_MPa]
        velocities = [self.velocity(P1) for P1 in P1_range_MPa]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # График массового расхода
        ax1.plot(P1_range_MPa, mass_flows, 'b-', linewidth=2.5)
        ax1.set_xlabel('Давление в трубе [МПа]', fontsize=12)
        ax1.set_ylabel('Массовый расход [кг/с]', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title(f'Зависимость массового расхода от давления\nДиаметр отверстия: {D_mm} мм',
                      fontsize=14, pad=20)

        # График скорости
        ax2.plot(P1_range_MPa, velocities, 'r-', linewidth=2.5)
        ax2.set_xlabel('Давление в трубе [МПа]', fontsize=12)
        ax2.set_ylabel('Скорость истечения [м/с]', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Зависимость скорости истечения от давления',
                      fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig('pressure_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diameter_dependence(self, P1_MPa, D_range):
        """
        Построение графика зависимости расхода от диаметра отверстия

        Parameters:
        P1_MPa (float): Давление в трубе [МПа]
        D_range (array): Массив значений диаметра [мм]
        """
        plt.style.use('default')
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5

        mass_flows = [self.mass_flow_rate(P1_MPa, D) for D in D_range]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(D_range, mass_flows, color='#2ca02c', linewidth=2.5)
        ax.set_xlabel('Диаметр отверстия [мм]', fontsize=12)
        ax.set_ylabel('Массовый расход [кг/с]', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'Зависимость массового расхода от диаметра отверстия\nДавление в трубе: {P1_MPa:.1f} МПа',
                     fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig('diameter_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()


# Пример использования
if __name__ == "__main__":
    # Свойства метана при нормальных условиях
    methane_properties = {
        'k': 1.32,  # Показатель адиабаты
        'R': 518.3,  # Газовая постоянная [Дж/(кг·К)]
        'T_C': 20,  # Температура [°C]
        'M': 0.016  # Молярная масса [кг/моль]
    }

    # Создание модели
    model = GasLeakModel(methane_properties)

    # Параметры утечки
    P1 = 1.0  # Давление в трубе [МПа]
    D = 10.0  # Диаметр отверстия [мм]

    # Расчет и вывод параметров
    pr_crit = model.critical_pressure_ratio()
    mass_flow = model.mass_flow_rate(P1, D)
    velocity = model.velocity(P1)

    print(f"Диаметр отверстия: {D:.1f} мм")
    print(f"Температура газа: {methane_properties['T_C']:.1f} °C")
    print(f"Давление в трубе: {P1:.1f} МПа")
    print(f"Атмосферное давление: {P_ATM / 1000:.1f} кПа")
    print(f"Критическое отношение давлений: {pr_crit:.3f}")
    print(f"Текущее отношение давлений P2/P1: {(P_ATM / (P1 * 1e6)):.3f}")
    print(f"Массовый расход газа: {mass_flow:.4f} кг/с")
    print(f"Скорость истечения: {velocity:.1f} м/с")

    # Построение графиков
    # График зависимости от давления
    P1_range = np.linspace(0.11, 2.0, 100)  # Давление от 0.11 до 2 МПа
    model.plot_pressure_dependence(D, P1_range)

    # График зависимости от диаметра
    D_range = np.linspace(1, 10, 50)
    model.plot_diameter_dependence(P1, D_range)
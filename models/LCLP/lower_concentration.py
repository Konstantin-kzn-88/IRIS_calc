import matplotlib

matplotlib.use('Agg')  # Установка бэкенда до импорта pyplot

from typing import List, Tuple
from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional


@dataclass
class LCLPResult:
    """Класс для хранения результатов расчета.

    Attributes:
        r_lclp (float): Радиус НКПР, м
        r_flash (float): Радиус пожара-вспышки, м
    """
    r_lclp: float
    r_flash: float


class InvalidParameterError(ValueError):
    """Исключение для некорректных входных параметров."""
    pass


class LCLP:
    """
    Класс для расчета зон НКПР и пожара-вспышки для паров ЛВЖ.

    Реализует методику расчета согласно Приказу МЧС № 404 от 10.07.2009.
    """

    # Константы для расчетов
    MOLAR_VOLUME: float = 22.413  # Молярный объем газа, л/моль
    TEMP_COEFFICIENT: float = 0.00367  # Температурный коэффициент
    RADIUS_COEFFICIENT: float = 7.8  # Коэффициент для расчета радиуса
    FLASH_FACTOR: float = 1.2  # Коэффициент для расчета радиуса пожара-вспышки

    def __init__(self):
        """Инициализация объекта класса."""
        # Настройка стиля для презентации
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'figure.figsize': [12, 8],
            'figure.dpi': 150,
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'legend.fontsize': 12,
            'legend.framealpha': 0.8,
            'lines.linewidth': 2.5
        })

    @staticmethod
    def _validate_inputs(mass: float, molecular_weight: float,
                         t_boiling: float, lower_concentration: float) -> None:
        """
        Проверка корректности входных параметров.

        Args:
            mass: Масса паров ЛВЖ/ГГ
            molecular_weight: Молекулярная масса
            t_boiling: Температура кипения
            lower_concentration: Нижний концентрационный предел

        Raises:
            InvalidParameterError: если параметры некорректны
        """
        if any(param <= 0 for param in [mass, molecular_weight, lower_concentration]):
            raise InvalidParameterError(
                'Масса, молекулярная масса и НКПР должны быть положительными числами'
            )

        if lower_concentration >= 100:
            raise InvalidParameterError(
                'Нижний концентрационный предел должен быть меньше 100%'
            )

        if not isinstance(t_boiling, (int, float)):
            raise InvalidParameterError(
                'Температура кипения должна быть числом'
            )

    def _calculate_vapour_density(self, molecular_weight: float, t_boiling: float) -> float:
        """
        Расчет плотности паров.

        Args:
            molecular_weight: Молекулярная масса, кг/кмоль
            t_boiling: Температура кипения, °C

        Returns:
            float: Плотность паров, кг/м³
        """
        return molecular_weight / (self.MOLAR_VOLUME * (1 + self.TEMP_COEFFICIENT * t_boiling))

    def lower_concentration_limit(self,
                                  mass: float,
                                  molecular_weight: float,
                                  t_boiling: float,
                                  lower_concentration: float) -> LCLPResult:
        """
        Расчет зон НКПР и пожара-вспышки для паров ЛВЖ.

        Args:
            mass: Масса паров ЛВЖ и ГГ, кг
            molecular_weight: Молекулярная масса, кг/кмоль
            t_boiling: Температура кипения, °C
            lower_concentration: Нижний концентрационный предел, % об.

        Returns:
            LCLPResult: Объект с результатами расчета
        """
        try:
            # Валидация входных параметров
            self._validate_inputs(mass, molecular_weight, t_boiling, lower_concentration)

            # Расчет плотности паров
            vapour_density = self._calculate_vapour_density(molecular_weight, t_boiling)

            # Расчет радиуса НКПР
            r_lclp = self.RADIUS_COEFFICIENT * math.pow(
                mass / (vapour_density * lower_concentration),
                1 / 3
            )

            # Расчет радиуса пожара-вспышки
            r_flash = r_lclp * self.FLASH_FACTOR

            # Округление результатов
            return LCLPResult(
                r_lclp=round(r_lclp, 2),
                r_flash=round(r_flash, 2)
            )

        except Exception as e:
            raise InvalidParameterError(f'Ошибка в расчете: {str(e)}')

    def plot_mass_dependency(self,
                             mass_range: Tuple[float, float],
                             molecular_weight: float,
                             t_boiling: float,
                             lower_concentration: float,
                             num_points: int = 50) -> Figure:
        """
        Построение графика зависимости радиусов от массы вещества.

        Args:
            mass_range: Диапазон масс (мин, макс), кг
            molecular_weight: Молекулярная масса, кг/кмоль
            t_boiling: Температура кипения, °C
            lower_concentration: Нижний концентрационный предел, % об.
            num_points: Количество точек для построения графика

        Returns:
            Figure: Объект графика matplotlib
        """
        masses = np.linspace(mass_range[0], mass_range[1], num_points)
        r_lclp_values = []
        r_flash_values = []

        for mass in masses:
            result = self.lower_concentration_limit(
                mass, molecular_weight, t_boiling, lower_concentration
            )
            r_lclp_values.append(result.r_lclp)
            r_flash_values.append(result.r_flash)

        # Создание графика с улучшенным дизайном для презентации
        fig, ax = plt.subplots()

        # Построение графиков с улучшенными цветами
        ax.plot(masses, r_lclp_values, color='#2E86C1', label='Радиус НКПР',
                linestyle='-', marker='o', markevery=5, markersize=8)
        ax.plot(masses, r_flash_values, color='#E74C3C', label='Радиус пожара-вспышки',
                linestyle='--', marker='s', markevery=5, markersize=8)

        # Настройка осей
        ax.set_xlabel('Масса вещества, кг')
        ax.set_ylabel('Радиус, м')

        # Настройка заголовка
        ax.set_title('Зависимость радиусов НКПР и пожара-вспышки\nот массы вещества', pad=20)

        # Улучшенная сетка
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)
        ax.minorticks_on()

        # Улучшенная легенда
        ax.legend(loc='upper left', fancybox=True, shadow=True)

        # Добавление подложки для улучшения читаемости
        ax.set_facecolor('#F8F9F9')

        # Настройка отступов
        plt.tight_layout()

        return fig

    def plot_concentration_dependency(self,
                                      mass: float,
                                      molecular_weight: float,
                                      t_boiling: float,
                                      conc_range: Tuple[float, float],
                                      num_points: int = 50) -> Figure:
        """
        Построение графика зависимости радиусов от концентрации.

        Args:
            mass: Масса паров ЛВЖ и ГГ, кг
            molecular_weight: Молекулярная масса, кг/кмоль
            t_boiling: Температура кипения, °C
            conc_range: Диапазон концентраций (мин, макс), % об.
            num_points: Количество точек для построения графика

        Returns:
            Figure: Объект графика matplotlib
        """
        concentrations = np.linspace(conc_range[0], conc_range[1], num_points)
        r_lclp_values = []
        r_flash_values = []

        for conc in concentrations:
            result = self.lower_concentration_limit(
                mass, molecular_weight, t_boiling, conc
            )
            r_lclp_values.append(result.r_lclp)
            r_flash_values.append(result.r_flash)

        # Создание графика с улучшенным дизайном для презентации
        fig, ax = plt.subplots()

        # Построение графиков с улучшенными цветами
        ax.plot(concentrations, r_lclp_values, color='#2E86C1', label='Радиус НКПР',
                linestyle='-', marker='o', markevery=5, markersize=8)
        ax.plot(concentrations, r_flash_values, color='#E74C3C', label='Радиус пожара-вспышки',
                linestyle='--', marker='s', markevery=5, markersize=8)

        # Настройка осей
        ax.set_xlabel('Концентрация, % об.')
        ax.set_ylabel('Радиус, м')

        # Настройка заголовка
        ax.set_title('Зависимость радиусов НКПР и пожара-вспышки\nот концентрации вещества', pad=20)

        # Улучшенная сетка
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.1)
        ax.minorticks_on()

        # Улучшенная легенда
        ax.legend(loc='upper right', fancybox=True, shadow=True)

        # Добавление подложки для улучшения читаемости
        ax.set_facecolor('#F8F9F9')

        # Настройка отступов
        plt.tight_layout()

        return fig


def main():
    """Пример использования класса LCLP с построением графиков."""
    try:
        # Создание экземпляра класса
        calculator = LCLP()

        # Входные данные
        test_data = {
            'mass': 10.96,  # кг
            'molecular_weight': 172.3,  # кг/кмоль
            't_boiling': 180,  # °C
            'lower_concentration': 3  # % об.
        }

        # Расчет
        result = calculator.lower_concentration_limit(**test_data)

        # Вывод результатов
        print(f'Результаты расчета:')
        print(f'Радиус НКПР: {result.r_lclp} м')
        print(f'Радиус пожара-вспышки: {result.r_flash} м')

        # Построение графиков
        # График зависимости от массы
        fig1 = calculator.plot_mass_dependency(
            mass_range=(1, 50),
            molecular_weight=test_data['molecular_weight'],
            t_boiling=test_data['t_boiling'],
            lower_concentration=test_data['lower_concentration']
        )
        # Сохранение графика с высоким разрешением
        fig1.savefig('mass_dependency.png',
                     dpi=300,
                     bbox_inches='tight',
                     facecolor='white',
                     edgecolor='none')

        # График зависимости от концентрации
        fig2 = calculator.plot_concentration_dependency(
            mass=test_data['mass'],
            molecular_weight=test_data['molecular_weight'],
            t_boiling=test_data['t_boiling'],
            conc_range=(0.5, 10)
        )
        # Сохранение графика с высоким разрешением
        fig2.savefig('concentration_dependency.png',
                     dpi=300,
                     bbox_inches='tight',
                     facecolor='white',
                     edgecolor='none')

    except InvalidParameterError as e:
        print(f'Ошибка в параметрах: {e}')
    except Exception as e:
        print(f'Неожиданная ошибка: {e}')


if __name__ == '__main__':
    main()
import math
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'pipeline_leak_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


class FluidType(Enum):
    """Перечисление типов жидкостей с их свойствами"""
    WATER = {
        "density": 998.0,
        "kinematic_viscosity": 1.006e-6,
        "name": "Вода"
    }
    OIL = {
        "density": 920.0,
        "kinematic_viscosity": 5e-5,
        "name": "Нефть"
    }
    GASOLINE = {
        "density": 750.0,
        "kinematic_viscosity": 0.5e-6,
        "name": "Бензин"
    }


@dataclass
class PipeParameters:
    """Класс для хранения параметров трубопровода"""
    diameter: float  # метры
    length: float  # метры
    roughness: float = 0.0001  # метры

    def validate(self) -> None:
        """Проверка корректности параметров трубопровода"""
        if self.diameter <= 0:
            raise ValueError("Диаметр трубы должен быть положительным числом")
        if self.length <= 0:
            raise ValueError("Длина трубы должна быть положительным числом")
        if self.roughness < 0:
            raise ValueError("Шероховатость не может быть отрицательной")
        if self.roughness > self.diameter / 2:
            raise ValueError("Шероховатость не может быть больше радиуса трубы")


@dataclass
class LeakParameters:
    """Класс для хранения параметров утечки"""
    pressure_start: float  # МПа
    pressure_end: float  # МПа
    height_diff: float  # метры
    hole_diameter: float  # метры
    discharge_coef: float = 0.62

    def validate(self) -> None:
        """Проверка корректности параметров утечки"""
        if self.pressure_start < 0:
            raise ValueError("Начальное давление не может быть отрицательным")
        if self.pressure_end < 0:
            raise ValueError("Конечное давление не может быть отрицательным")
        if self.pressure_end > self.pressure_start:
            raise ValueError("Конечное давление не может быть больше начального")
        if self.discharge_coef <= 0 or self.discharge_coef > 1:
            raise ValueError("Коэффициент расхода должен быть в диапазоне (0,1]")


def mpa_to_pa(pressure_mpa: float) -> float:
    """Конвертация давления из МПа в Па"""
    return pressure_mpa * 1e6


def pa_to_mpa(pressure_pa: float) -> float:
    """Конвертация давления из Па в МПа"""
    return pressure_pa / 1e6


def calculate_reynolds_number(
        velocity: float,
        diameter: float,
        kinematic_viscosity: float
) -> float:
    """
    Расчет числа Рейнольдса

    Args:
        velocity: Скорость потока, м/с
        diameter: Диаметр трубы, м
        kinematic_viscosity: Кинематическая вязкость, м²/с

    Returns:
        float: Число Рейнольдса
    """
    return velocity * diameter / kinematic_viscosity


def calculate_friction_factor(
        reynolds: float,
        relative_roughness: float,
        max_iterations: int = 100,
        tolerance: float = 1e-6
) -> float:
    """
    Расчет коэффициента трения по формуле Колбрука-Уайта с оптимизированной итерацией

    Args:
        reynolds: Число Рейнольдса
        relative_roughness: Относительная шероховатость
        max_iterations: Максимальное число итераций
        tolerance: Допустимая погрешность

    Returns:
        float: Коэффициент трения
    """
    if reynolds < 2300:
        return 64 / reynolds

    def colebrook_white(f: float) -> float:
        return -2 * math.log10(relative_roughness / 3.7 + 2.51 / (reynolds * math.sqrt(f)))

    # Начальное приближение по явной формуле Свэмми-Джейна
    f = 0.25 / (math.log10(relative_roughness / 3.7 + 5.74 / reynolds ** 0.9)) ** 2

    for _ in range(max_iterations):
        f_new = 1 / colebrook_white(f) ** 2
        if abs(f_new - f) < tolerance:
            return f_new
        f = f_new

    logger.warning("Достигнуто максимальное число итераций при расчете коэффициента трения")
    return f


class LeakCalculator:
    """Класс для расчета параметров утечки"""

    def __init__(
            self,
            pipe: PipeParameters,
            leak: LeakParameters,
            fluid_type: FluidType = FluidType.WATER
    ):
        self.pipe = pipe
        self.leak = leak
        self.fluid_type = fluid_type
        self.g = 9.81  # ускорение свободного падения, м/с²

        # Валидация параметров
        self.pipe.validate()
        self.leak.validate()

        # Расчет базовых параметров
        self.pipe_area = math.pi * (self.pipe.diameter / 2) ** 2
        self.hole_area = math.pi * (self.leak.hole_diameter / 2) ** 2
        self.pipe_volume = self.pipe_area * self.pipe.length

        logger.info(f"Инициализация расчета утечки для {fluid_type.value['name']}")

    def calculate_initial_state(self) -> Dict[str, float]:
        """Расчет начального состояния утечки"""
        try:
            # Конвертация давлений
            p_start = mpa_to_pa(self.leak.pressure_start)
            p_end = mpa_to_pa(self.leak.pressure_end)

            # Расчет напора
            pressure_head = (p_start - p_end) / (self.fluid_type.value['density'] * self.g)
            total_head = pressure_head + self.leak.height_diff

            # Расчет скорости истечения
            velocity = self.leak.discharge_coef * math.sqrt(2 * self.g * abs(total_head))

            # Расчет расходов
            volume_flow = velocity * self.hole_area
            mass_flow = volume_flow * self.fluid_type.value['density']

            # Расчет числа Рейнольдса
            reynolds = calculate_reynolds_number(
                velocity,
                self.pipe.diameter,
                self.fluid_type.value['kinematic_viscosity']
            )

            # Расчет коэффициента трения
            relative_roughness = self.pipe.roughness / self.pipe.diameter
            friction_factor = calculate_friction_factor(reynolds, relative_roughness)

            # Расчет потерь давления
            velocity_pipe = volume_flow / self.pipe_area
            pressure_drop = (
                    friction_factor *
                    self.pipe.length *
                    self.fluid_type.value['density'] *
                    velocity_pipe ** 2 /
                    (2 * self.pipe.diameter)
            )

            # Расчет времени опорожнения
            if self.leak.hole_diameter < self.pipe.diameter:
                emptying_time = self.pipe_volume / (self.hole_area * velocity)
            else:
                emptying_time = math.sqrt(2 * self.pipe.length / self.g)

            # Расчет энергетических характеристик
            kinetic_energy = 0.5 * self.fluid_type.value['density'] * velocity ** 2
            potential_energy = self.fluid_type.value['density'] * self.g * abs(self.leak.height_diff)
            total_energy = kinetic_energy + potential_energy

            results = {
                "velocity": velocity,
                "volume_flow": volume_flow,
                "volume_flow_hour": volume_flow * 3600,
                "mass_flow": mass_flow,
                "emptying_time": emptying_time,
                "emptying_time_min": emptying_time / 60,
                "reynolds": reynolds,
                "total_head": total_head,
                "pressure_drop_mpa": pa_to_mpa(pressure_drop),
                "effective_pressure_mpa": self.leak.pressure_start - pa_to_mpa(pressure_drop),
                "friction_factor": friction_factor,
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
                "total_energy": total_energy
            }

            logger.info("Успешно рассчитано начальное состояние утечки")
            return results

        except Exception as e:
            logger.error(f"Ошибка при расчете начального состояния: {str(e)}")
            raise

    def calculate_time_series(
            self,
            time_steps: int = 100
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Расчет изменения параметров истечения во времени с учетом энергетических характеристик
        """
        try:
            # Расчет базового времени опорожнения
            initial_state = self.calculate_initial_state()
            emptying_time = initial_state["emptying_time"]

            # Создание временной шкалы
            time_array = np.linspace(0, emptying_time, time_steps)

            # Инициализация массивов результатов
            results = {
                "mass_flow": np.zeros(time_steps),
                "pressure": np.zeros(time_steps),
                "volume": np.zeros(time_steps),
                "velocity": np.zeros(time_steps),
                "kinetic_energy": np.zeros(time_steps),
                "potential_energy": np.zeros(time_steps),
                "total_energy": np.zeros(time_steps)
            }

            # Расчет параметров для каждого момента времени
            for i, t in enumerate(time_array):
                # Оставшийся объем жидкости
                remaining_volume = max(0, self.pipe_volume * (1 - t / emptying_time))
                results["volume"][i] = remaining_volume

                # Текущее давление
                current_pressure_pa = mpa_to_pa(self.leak.pressure_start) * (remaining_volume / self.pipe_volume)
                results["pressure"][i] = pa_to_mpa(current_pressure_pa)

                # Расчет напора
                current_head = (
                        current_pressure_pa / (self.fluid_type.value['density'] * self.g) +
                        self.leak.height_diff
                )

                # Скорость истечения
                velocity = self.leak.discharge_coef * math.sqrt(2 * self.g * abs(current_head))
                results["velocity"][i] = velocity

                # Массовый расход
                results["mass_flow"][i] = velocity * self.hole_area * self.fluid_type.value['density']

                # Энергетические характеристики
                results["kinetic_energy"][i] = 0.5 * self.fluid_type.value['density'] * velocity ** 2
                results["potential_energy"][i] = (
                        self.fluid_type.value['density'] *
                        self.g *
                        abs(self.leak.height_diff)
                )
                results["total_energy"][i] = (
                        results["kinetic_energy"][i] +
                        results["potential_energy"][i]
                )

            logger.info("Успешно рассчитаны временные ряды параметров утечки")
            return time_array, results

        except Exception as e:
            logger.error(f"Ошибка при расчете временных рядов: {str(e)}")
            raise

    def plot_results(
            self,
            time_array: np.ndarray,
            results: Dict[str, np.ndarray],
            save_path: str = 'leak_analysis.png'
    ) -> None:
        """Построение расширенных графиков результатов анализа"""
        try:
            # Настройка стиля графиков
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.linestyle'] = '--'
            plt.rcParams['grid.alpha'] = 0.5
            plt.rcParams['axes.axisbelow'] = True  # Сетка под графиком
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))

            # График массового расхода
            axes[0].plot(
                time_array / 60,
                results["mass_flow"],
                'b-',
                linewidth=2,
                label='Массовый расход'
            )
            axes[0].set_xlabel('Время, мин')
            axes[0].set_ylabel('Массовый расход, кг/с')
            axes[0].set_title('Изменение массового расхода во времени')
            axes[0].legend(loc='best')
            axes[0].grid(True)

            # График давления
            axes[1].plot(
                time_array / 60,
                results["pressure"],
                'r-',
                linewidth=2,
                label='Давление'
            )
            axes[1].set_xlabel('Время, мин')
            axes[1].set_ylabel('Давление, МПа')
            axes[1].set_title('Изменение давления во времени')
            axes[1].legend(loc='best')
            axes[1].grid(True)

            # График энергетических характеристик
            axes[2].plot(
                time_array / 60,
                results["kinetic_energy"],
                'g-',
                linewidth=2,
                label='Кинетическая энергия'
            )
            axes[2].plot(
                time_array / 60,
                results["potential_energy"],
                'm-',
                linewidth=2,
                label='Потенциальная энергия'
            )
            axes[2].plot(
                time_array / 60,
                results["total_energy"],
                'k--',
                linewidth=2,
                label='Полная энергия'
            )
            axes[2].set_xlabel('Время, мин')
            axes[2].set_ylabel('Энергия, Дж/кг')
            axes[2].set_title('Изменение энергетических характеристик во времени')
            axes[2].legend(loc='best')
            axes[2].grid(True)

            # Добавление информации о параметрах
            info_text = (
                f'Параметры расчета:\n'
                f'Тип жидкости: {self.fluid_type.value["name"]}\n'
                f'Начальное давление: {self.leak.pressure_start:.1f} МПа\n'
                f'Длина трубопровода: {self.pipe.length:.0f} м\n'
                f'Диаметр трубы: {self.pipe.diameter * 1000:.0f} мм\n'
                f'Диаметр отверстия: {self.leak.hole_diameter * 1000:.0f} мм\n'
                f'Разница высот: {self.leak.height_diff:.1f} м\n'
                f'Плотность жидкости: {self.fluid_type.value["density"]:.0f} кг/м³'
            )

            fig.text(
                0.05, 0.02,
                info_text,
                fontsize=10,
                family='monospace',
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='bottom',
                horizontalalignment='left'
            )

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Графики сохранены в файл: {save_path}")

        except Exception as e:
            logger.error(f"Ошибка при построении графиков: {str(e)}")
            raise


def create_report(
        calculator: LeakCalculator,
        initial_state: Dict[str, float],
        save_path: str = 'leak_report.txt'
) -> None:
    """
    Создание подробного отчета о результатах расчета

    Args:
        calculator: Экземпляр калькулятора утечки
        initial_state: Словарь с начальными значениями параметров
        save_path: Путь для сохранения отчета
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("ОТЧЕТ ПО АНАЛИЗУ УТЕЧКИ\n")
            f.write("=" * 50 + "\n\n")

            # Информация о жидкости
            f.write("ПАРАМЕТРЫ ЖИДКОСТИ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Тип жидкости: {calculator.fluid_type.value['name']}\n")
            f.write(f"Плотность: {calculator.fluid_type.value['density']:.1f} кг/м³\n")
            f.write(f"Кинематическая вязкость: {calculator.fluid_type.value['kinematic_viscosity']:.2e} м²/с\n\n")

            # Параметры трубопровода
            f.write("ПАРАМЕТРЫ ТРУБОПРОВОДА:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Длина: {calculator.pipe.length:.1f} м\n")
            f.write(f"Диаметр: {calculator.pipe.diameter * 1000:.1f} мм\n")
            f.write(f"Шероховатость: {calculator.pipe.roughness * 1000:.2f} мм\n")
            f.write(f"Объем участка: {calculator.pipe_volume:.1f} м³\n\n")

            # Параметры утечки
            f.write("ПАРАМЕТРЫ УТЕЧКИ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Начальное давление: {calculator.leak.pressure_start:.2f} МПа\n")
            f.write(f"Конечное давление: {calculator.leak.pressure_end:.2f} МПа\n")
            f.write(f"Разница высот: {calculator.leak.height_diff:.1f} м\n")
            f.write(f"Диаметр отверстия: {calculator.leak.hole_diameter * 1000:.1f} мм\n")
            f.write(f"Коэффициент расхода: {calculator.leak.discharge_coef:.2f}\n\n")

            # Результаты расчета
            f.write("РЕЗУЛЬТАТЫ РАСЧЕТА:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Скорость истечения: {initial_state['velocity']:.2f} м/с\n")
            f.write(f"Объемный расход: {initial_state['volume_flow_hour']:.2f} м³/ч\n")
            f.write(f"Массовый расход: {initial_state['mass_flow']:.2f} кг/с\n")
            f.write(f"Время опорожнения: {initial_state['emptying_time_min']:.1f} мин\n")
            f.write(f"Число Рейнольдса: {initial_state['reynolds']:.0f}\n")
            f.write(f"Коэффициент трения: {initial_state['friction_factor']:.4f}\n")
            f.write(f"Потери давления: {initial_state['pressure_drop_mpa']:.3f} МПа\n")
            f.write(f"Эффективное давление: {initial_state['effective_pressure_mpa']:.3f} МПа\n\n")

            # Энергетические характеристики
            f.write("ЭНЕРГЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Кинетическая энергия: {initial_state['kinetic_energy']:.1f} Дж/кг\n")
            f.write(f"Потенциальная энергия: {initial_state['potential_energy']:.1f} Дж/кг\n")
            f.write(f"Полная энергия: {initial_state['total_energy']:.1f} Дж/кг\n")

        logger.info(f"Отчет сохранен в файл: {save_path}")

    except Exception as e:
        logger.error(f"Ошибка при создании отчета: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Создание параметров трубопровода
        pipe = PipeParameters(
            diameter=0.1,  # 100 мм
            length=1000,  # 1000 м
            roughness=0.0001  # 0.1 мм
        )

        # Создание параметров утечки
        leak = LeakParameters(
            pressure_start=5.0,  # 5 МПа
            pressure_end=0.101325,  # атмосферное давление
            height_diff=-2,  # утечка ниже уровня сравнения на 2 м
            hole_diameter=0.01  # 10 мм
        )

        # Создание калькулятора с выбором типа жидкости
        calculator = LeakCalculator(pipe, leak, FluidType.WATER)

        # Расчет начального состояния
        initial_state = calculator.calculate_initial_state()

        # Расчет временных рядов
        time_array, results = calculator.calculate_time_series()

        # Построение графиков
        calculator.plot_results(time_array, results, 'leak_analysis.png')

        # Создание отчета
        create_report(calculator, initial_state, 'leak_report.txt')

        logger.info("Расчет успешно завершен")

    except Exception as e:
        logger.error(f"Ошибка при выполнении расчета: {str(e)}")
        raise
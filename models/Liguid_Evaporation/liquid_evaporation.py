import numpy as np
from scipy.integrate import odeint
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhysicalConstants:
    """Физические константы, используемые в модели"""
    R: float = 8.314  # Универсальная газовая постоянная [Дж/(моль·K)]
    P0: float = 101325  # Стандартное давление [Па]
    DEFAULT_CP: float = 4180  # Удельная теплоемкость воды [Дж/(кг·K)]
    MIN_TEMP_C: float = 0  # Минимальная температура [°C]
    MAX_TEMP_C: float = 100  # Максимальная температура [°C]

    @property
    def MIN_TEMP(self) -> float:
        """Минимальная температура в Кельвинах"""
        return self.MIN_TEMP_C + 273.15

    @property
    def MAX_TEMP(self) -> float:
        """Максимальная температура в Кельвинах"""
        return self.MAX_TEMP_C + 273.15


class ValidationError(Exception):
    """Исключение для ошибок валидации параметров"""
    pass


class PhysicalConstraintError(Exception):
    """Исключение для нарушения физических ограничений"""
    pass


class EvaporationModel:
    def __init__(self, params: Dict[str, float], constants: Optional[PhysicalConstants] = None):
        """
        Инициализация модели испарения

        Args:
            params: Словарь параметров, где температуры задаются в градусах Цельсия
            constants: Физические константы (опционально)
        """
        # Конвертация температур из Цельсия в Кельвины
        params_k = self._convert_temps_to_kelvin(params)
        self._validate_params(params_k)
        self.params = params_k
        self.constants = constants or PhysicalConstants()
        logger.info("Модель испарения инициализирована с параметрами: %s", params)

    @staticmethod
    def _convert_temps_to_kelvin(params: Dict[str, float]) -> Dict[str, float]:
        """Конвертация температурных параметров из Цельсия в Кельвины"""
        temp_params = ['T_boil', 'T_init', 'T_amb', 'T_ground']
        params_k = params.copy()

        for param in temp_params:
            if param in params_k:
                params_k[param] = params_k[param] + 273.15

        return params_k

    @staticmethod
    def _validate_params(params: Dict[str, float]) -> None:
        """Валидация входных параметров"""
        required_params = {
            'M': 'молекулярная масса [кг/моль]',
            'L': 'удельная теплота испарения [Дж/кг]',
            'T_boil': 'температура кипения [K]',
            'T_init': 'начальная температура [K]',
            'T_amb': 'температура окружающей среды [K]',
            'T_ground': 'температура поверхности [K]',
            'wind_speed': 'скорость ветра [м/с]',
            'solar_flux': 'поток солнечного излучения [Вт/м²]',
            'initial_mass': 'начальная масса [кг]',
            'spill_area': 'площадь пролива [м²]'
        }

        for param, description in required_params.items():
            if param not in params:
                raise ValidationError(f"Отсутствует обязательный параметр: {param} ({description})")
            if params[param] <= 0:
                raise ValidationError(f"Параметр {param} должен быть положительным")

    def vapor_pressure(self, T: float) -> float:
        """Расчет давления насыщенных паров"""
        T = np.clip(T, self.constants.MIN_TEMP, self.constants.MAX_TEMP)
        try:
            P = self.constants.P0 * np.exp((self.params['L'] * self.params['M'] / self.constants.R) *
                                           (1 / self.params['T_boil'] - 1 / T))
            return min(P, self.constants.P0)
        except (ZeroDivisionError, OverflowError):
            return 0.0

    def mass_transfer_coefficient(self) -> float:
        """Расчет коэффициента массопереноса"""
        Re = min(self.params['wind_speed'] * np.sqrt(self.params['spill_area']) / 1.5e-5, 1e7)
        Sc = 0.7

        if Re < 5e5:
            Sh = 0.664 * np.sqrt(Re) * np.power(Sc, 0.33)
        else:
            Sh = 0.037 * np.power(Re, 0.8) * np.power(Sc, 0.33)

        return min(Sh * 1.5e-5 / np.sqrt(self.params['spill_area']), 0.01)

    def heat_transfer_coefficient(self, T: float) -> float:
        """
        Расчет коэффициента теплопередачи с учетом вынужденной и естественной конвекции

        Args:
            T: Текущая температура жидкости [K]
        Returns:
            float: Коэффициент теплопередачи [Вт/(м²·K)]
        """
        h_forced = 5.7 + 3.8 * self.params['wind_speed']
        dT = abs(self.params['T_amb'] - T)
        h_natural = 1.31 * np.power(dT, 1 / 3)
        h_total = np.power(h_forced ** 3 + h_natural ** 3, 1 / 3)
        return min(h_total, 100)

    def evaporation_rate(self, T: float) -> float:
        """Расчет скорости испарения"""
        T = np.clip(T, self.constants.MIN_TEMP, self.constants.MAX_TEMP)
        k_g = self.mass_transfer_coefficient()
        P_vap = self.vapor_pressure(T)

        try:
            rate = k_g * self.params['M'] * P_vap / (self.constants.R * T)
            max_rate = self.params['initial_mass'] / (3600 * self.params['spill_area'])
            return min(rate, max_rate)
        except (ZeroDivisionError, OverflowError):
            return 0.0

    def energy_balance(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Уравнение энергетического баланса с учетом всех тепловых потоков

        Args:
            state: Массив состояния [T, m]
            t: Время [с]
        Returns:
            np.ndarray: Производные [dT/dt, dm/dt]
        """
        T, m = state

        T = np.clip(T, self.constants.MIN_TEMP, self.constants.MAX_TEMP)
        if m <= 0:
            return np.array([0.0, 0.0])

        try:
            alpha_solar = 0.1
            q_solar = min(self.params['solar_flux'] * self.params['spill_area'] * alpha_solar, 1000)

            h_air = self.heat_transfer_coefficient(T)
            q_conv = h_air * self.params['spill_area'] * (self.params['T_amb'] - T)

            h_ground = h_air * 0.5
            q_ground = h_ground * self.params['spill_area'] * (self.params['T_ground'] - T)

            dm_dt = -self.evaporation_rate(T) * self.params['spill_area']
            dm_dt = max(dm_dt, -m / 100)
            q_evap = dm_dt * self.params['L']

            q_total = q_solar + q_conv + q_ground + q_evap
            dT_dt = q_total / (m * self.constants.DEFAULT_CP)
            dT_dt = np.clip(dT_dt, -2, 2)

            return np.array([dT_dt, dm_dt])

        except (ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Ошибка в расчете теплового баланса: {str(e)}")
            return np.array([0.0, 0.0])

    def simulate(self, t_span: float, dt: float) -> Dict[str, np.ndarray]:
        """Моделирование процесса испарения"""
        t = np.arange(0, t_span, dt)
        initial_state = np.array([self.params['T_init'], self.params['initial_mass']])

        try:
            solution = odeint(
                self.energy_balance,
                initial_state,
                t,
                full_output=True,
                rtol=1e-6,
                atol=1e-6,
                mxstep=5000
            )[0]

            solution[:, 0] = np.clip(solution[:, 0],
                                     self.constants.MIN_TEMP,
                                     self.constants.MAX_TEMP)
            solution[:, 1] = np.maximum(solution[:, 1], 0)

            evap_rate = np.array([self.evaporation_rate(T) * self.params['spill_area']
                                  for T in solution[:, 0]])

            results = {
                't': t,
                'T': solution[:, 0],
                'm': solution[:, 1],
                'evap_rate': -evap_rate
            }

            results.update(self.analyze_results(results))
            return results

        except Exception as e:
            logger.error(f"Ошибка при моделировании: {str(e)}")
            raise

    def analyze_results(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Анализ результатов моделирования"""
        try:
            initial_mass = self.params['initial_mass']
            final_mass = float(results['m'][-1])

            if final_mass < 0:
                final_mass = 0

            total_evaporated = initial_mass - final_mass

            evap_rates = np.abs(results['evap_rate'])
            average_rate = float(np.mean(evap_rates))
            max_rate = float(np.max(evap_rates))

            return {
                'total_evaporated': total_evaporated,
                'average_rate': average_rate,
                'max_rate': max_rate,
                'max_temp': float(np.max(results['T']))
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе результатов: {str(e)}")
            raise

    def plot_results(self, results: Dict[str, np.ndarray], save_path: Optional[str] = None) -> None:
        """Построение графиков результатов"""
        try:
            plt.style.use('default')

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

            t_minutes = results['t'] / 60

            # График температуры (конвертация в градусы Цельсия для отображения)
            ax1.plot(t_minutes, results['T'] - 273.15, 'b-', linewidth=2)
            ax1.set_xlabel('Время [мин]')
            ax1.set_ylabel('Температура [°C]')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_title('Изменение температуры жидкости')

            # Добавление исходных данных (отображение в градусах Цельсия)
            params_text = (
                f"Исходные данные:\n"
                f"Начальная масса: {self.params['initial_mass']:.1f} кг\n"
                f"Площадь пролива: {self.params['spill_area']:.1f} м²\n"
                f"Начальная температура: {self.params['T_init'] - 273.15:.1f}°C\n"
                f"Температура воздуха: {self.params['T_amb'] - 273.15:.1f}°C\n"
                f"Температура поверхности: {self.params['T_ground'] - 273.15:.1f}°C\n"
                f"Скорость ветра: {self.params['wind_speed']:.1f} м/с\n"
                f"Солнечный поток: {self.params['solar_flux']:.0f} Вт/м²"
            )

            ax1.text(0.98, 0.95, params_text,
                     transform=ax1.transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=8)

            # График массы
            ax2.plot(t_minutes, results['m'], 'g-', linewidth=2)
            ax2.set_xlabel('Время [мин]')
            ax2.set_ylabel('Масса [кг]')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_title('Изменение массы жидкости')

            # График скорости испарения
            ax3.plot(t_minutes, np.abs(results['evap_rate']), 'r-', linewidth=2)
            ax3.set_xlabel('Время [мин]')
            ax3.set_ylabel('Скорость испарения [кг/с]')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_title('Скорость испарения')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сохранен в {save_path}")
                plt.close(fig)

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графиков: {str(e)}")
            raise


# Пример использования
if __name__ == "__main__":
    # Параметры для воды (температуры в градусах Цельсия)
    params = {
        'M': 0.018,  # молекулярная масса [кг/моль]
        'L': 2.26e6,  # удельная теплота испарения [Дж/кг]
        'T_boil': 100,  # температура кипения [°C]
        'T_init': 180,  # начальная температура [°C]
        'T_amb': 25,  # температура окружающей среды [°C]
        'T_ground': 300,  # температура поверхности [°C]
        'wind_speed': 3.0,  # скорость ветра [м/с]
        'solar_flux': 500,  # поток солнечного излучения [Вт/м²]
        'initial_mass': 100,  # начальная масса [кг]
        'spill_area': 10  # площадь пролива [м²]
    }

    try:
        # Создание модели
        model = EvaporationModel(params)

        # Моделирование
        results = model.simulate(t_span=3600, dt=10)

        # Вывод результатов
        print("\nРезультаты моделирования:")
        print(f"Общая масса испарившейся жидкости: {results['total_evaporated']:.2f} кг")
        print(f"Средняя скорость испарения: {results['average_rate']:.4f} кг/с")
        print(f"Максимальная скорость испарения: {results['max_rate']:.4f} кг/с")
        print(f"Максимальная температура: {results['max_temp'] - 273.15:.1f} °C")

        # Построение графиков
        model.plot_results(results, save_path='evaporation_results.png')

    except ValidationError as e:
        logger.error(f"Ошибка валидации: {str(e)}")
    except PhysicalConstraintError as e:
        logger.error(f"Нарушение физических ограничений: {str(e)}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
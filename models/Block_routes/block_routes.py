import numpy as np
import matplotlib
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class ToxicityParams:
    """Параметры токсичности продуктов горения"""
    co_yield: float  # выход CO, кг/кг
    co2_yield: float  # выход CO2, кг/кг
    hcl_yield: float  # выход HCl, кг/кг


@dataclass
class LiquidFuelParams:
    """Параметры разлива жидкого топлива"""
    density: float  # плотность топлива, кг/м³
    thickness: float  # средняя толщина разлива, м
    spread_rate: float  # скорость растекания, м²/с


class FireHazardCalculator:
    def __init__(self, room_params: Dict, material_params: Dict,
                 toxic_params: Optional[ToxicityParams] = None,
                 liquid_params: Optional[LiquidFuelParams] = None):
        """
        Инициализация калькулятора пожарной опасности.

        Args:
            room_params: Параметры помещения
            material_params: Параметры горючего материала
            toxic_params: Параметры токсичности (опционально)
            liquid_params: Параметры жидкого топлива (опционально)
        """
        # Room parameters
        self.volume = room_params['volume']  # м³
        self.vent_area = room_params['vent_area']  # м²
        self.init_temp = room_params['init_temp']  # K
        self.init_burn_area = room_params['burn_area']  # м²
        self.room_height = room_params.get('height', 3.0)  # м
        self.floor_area = room_params.get('floor_area', self.volume / self.room_height)  # м²

        # Material parameters
        self.heat_value = material_params['heat_value']  # кДж/кг
        self.smoke_value = material_params['smoke_value']  # Нп×м²/кг
        self.burn_rate = material_params['burn_rate']  # кг/(м²×с)

        # Liquid fuel parameters
        self.liquid_params = liquid_params

        # Toxicity parameters
        self.toxic_params = toxic_params

        # Constants
        self.cp = 1.005  # удельная теплоемкость воздуха, кДж/(кг·К)
        self.rho = 1.29  # начальная плотность воздуха, кг/м³
        self.heat_loss = 0.3  # коэффициент потерь тепла (уменьшен для бензина)
        self.burn_eff = 0.97  # полнота сгорания (увеличена для бензина)

        # Critical values
        self.crit_temp = 343.15  # K (70°C)
        self.crit_visibility = 20  # м
        self.crit_co = 0.11  # кг/м³
        self.crit_co2 = 0.11  # кг/м³
        self.crit_hcl = 0.058  # кг/м³

    def calculate_burn_area(self, t: float) -> float:
        """
        Расчет площади горения с учетом растекания жидкости.

        Args:
            t: Время от начала пожара, с

        Returns:
            float: Площадь горения, м²
        """
        if not self.liquid_params:
            return self.init_burn_area

        # Расчет площади разлива
        spread_area = min(
            self.init_burn_area + self.liquid_params.spread_rate * t,
            self.floor_area  # Ограничение площадью помещения
        )

        return spread_area

    def burned_mass(self, t: float) -> float:
        """
        Расчет массы выгоревшего материала.

        Args:
            t: Время от начала пожара, с

        Returns:
            float: Масса выгоревшего материала, кг
        """
        if self.liquid_params:
            # Интегрирование по времени с учетом изменяющейся площади
            times = np.linspace(0, t, 100)
            areas = [self.calculate_burn_area(tau) for tau in times]
            dt = t / 99
            return sum(self.burn_rate * area * dt for area in areas)

        return self.burn_rate * self.init_burn_area * t

    def temperature(self, t: float) -> float:
        """
        Расчет температуры в помещении.

        Args:
            t: Время от начала пожара, с

        Returns:
            float: Температура в помещении, K
        """
        # Мощность тепловыделения с учетом изменяющейся площади
        burn_area = self.calculate_burn_area(t)
        Q = self.heat_value * self.burn_rate * burn_area * (1 - self.heat_loss) * self.burn_eff

        # Учет теплоемкости воздуха и ограждающих конструкций
        V_eff = self.volume * 1.3  # Уменьшен коэффициент для бензина

        # Коэффициент газообмена
        k_vent = np.sqrt(self.vent_area / self.volume) * 0.6  # Увеличен для бензина

        # Максимальная температура пожара (увеличена для бензина)
        T_max = 1373.15  # ~1100°C

        # Расчет приращения температуры с учетом ограничения роста
        dT_max = T_max - self.init_temp
        dT_current = (Q * t) / (self.cp * self.rho * V_eff) * (1 - np.exp(-k_vent * t))

        # Корректировка с учетом более быстрого роста для бензина
        dT = dT_max * (1 - np.exp(-1.2 * dT_current / dT_max))

        # Учет меньших теплопотерь при горении бензина
        heat_loss_factor = 1 + 0.1 * (dT / dT_max)
        dT = dT / heat_loss_factor

        return self.init_temp + dT

    def visibility(self, t: float) -> float:
        """Расчет потери видимости"""
        return self.smoke_value * self.burned_mass(t) / (self.volume * self.crit_visibility)

    def toxic_concentrations(self, t: float) -> Dict[str, float]:
        """Расчет концентраций токсичных веществ"""
        if not self.toxic_params:
            return {}

        burned = self.burned_mass(t)
        return {
            'CO': self.toxic_params.co_yield * burned / self.volume,
            'CO2': self.toxic_params.co2_yield * burned / self.volume,
            'HCl': self.toxic_params.hcl_yield * burned / self.volume
        }

    # Остальные методы остаются без изменений
    def calculate_blocking_time(self, t_max: float = 300, dt: float = 1) -> Dict:
        """Расчет времени блокирования по всем факторам"""
        times = np.arange(0, t_max, dt)
        result = {
            'temperature_time': None,
            'visibility_time': None,
            'co_time': None,
            'co2_time': None,
            'hcl_time': None
        }

        for t in times:
            temp = self.temperature(t)
            vis = self.visibility(t)

            if self.toxic_params:
                tox = self.toxic_concentrations(t)

                if result['co_time'] is None and tox['CO'] >= self.crit_co:
                    result['co_time'] = t
                if result['co2_time'] is None and tox['CO2'] >= self.crit_co2:
                    result['co2_time'] = t
                if result['hcl_time'] is None and tox['HCl'] >= self.crit_hcl:
                    result['hcl_time'] = t

            if result['temperature_time'] is None and temp >= self.crit_temp:
                result['temperature_time'] = t

            if result['visibility_time'] is None and vis >= 1.0:
                result['visibility_time'] = t

            all_found = all(
                (result[key] is not None if key != 'blocking_time' else True)
                for key in result.keys()
                if key not in ['co_time', 'co2_time', 'hcl_time']
                or (self.toxic_params and key == 'co_time')
                or (self.toxic_params and key == 'co2_time')
                or (self.toxic_params and key == 'hcl_time')
            )

            if all_found:
                break

        valid_times = [t for t in result.values() if t is not None]
        result['blocking_time'] = min(valid_times) if valid_times else None

        return result

    def save_results(self, results: Dict, filename: str = 'fire_results.json') -> None:
        """
        Сохранение результатов расчета в файл.

        Args:
            results: Результаты расчета
            filename: Имя файла для сохранения
        """
        # Конвертируем numpy типы в обычные Python типы
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)

    def plot_dynamics(self, t_max: float = 300, save_path: str = 'fire_dynamics.png') -> None:
        """Построение графиков динамики ОФП"""
        times = np.linspace(0, t_max, 300)
        temps = [self.temperature(t) - 273.15 for t in times]  # перевод в °C
        visibilities = [self.visibility(t) for t in times]
        n_plots = 3 if self.toxic_params else 3  # Увеличено количество графиков
        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        # График температуры
        axs[0].plot(times, temps, 'r-', label='Температура')
        axs[0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Критическое значение')
        axs[0].set_xlabel('Время, с')
        axs[0].set_ylabel('Температура, °C')
        axs[0].grid(True)
        axs[0].legend()

        # График задымления
        axs[1].plot(times, visibilities, 'b-', label='Задымление')
        axs[1].axhline(y=1.0, color='b', linestyle='--', alpha=0.5, label='Критическое значение')
        axs[1].set_xlabel('Время, с')
        axs[1].set_ylabel('Относительная задымленность')
        axs[1].grid(True)
        axs[1].legend()

        # График токсичных веществ
        if self.toxic_params:
            tox_data = [self.toxic_concentrations(t) for t in times]
            co_values = [d['CO'] for d in tox_data]
            co2_values = [d['CO2'] for d in tox_data]
            hcl_values = [d['HCl'] for d in tox_data]

            axs[2].plot(times, co_values, 'g-', label='CO')
            axs[2].plot(times, co2_values, 'm-', label='CO2')
            axs[2].plot(times, hcl_values, 'y-', label='HCl')
            axs[2].axhline(y=self.crit_co, color='g', linestyle='--', alpha=0.5)
            axs[2].axhline(y=self.crit_co2, color='m', linestyle='--', alpha=0.5)
            axs[2].axhline(y=self.crit_hcl, color='y', linestyle='--', alpha=0.5)
            axs[2].set_xlabel('Время, с')
            axs[2].set_ylabel('Концентрация, кг/м³')
            axs[2].grid(True)
            axs[2].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def main():
    # Параметры помещения
    room = {
        'volume': 20*30*6,  # м³
        'vent_area': 1,  # м²
        'init_temp': 293,  # K
        'burn_area': 1.0,  # м² (начальная площадь пролива)
        'height': 6.0,  # м
        'floor_area': 20*30  # м² (площадь пола помещения)
    }

    # Параметры бензина
    gasoline = {
        'heat_value': 44000,  # кДж/кг
        'smoke_value': 800,  # Нп×м²/кг
        'burn_rate': 0.06  # кг/(м²×с)
    }

    # Параметры токсичности для бензина
    toxic = ToxicityParams(
        co_yield=0.1094,  # кг/кг
        co2_yield=7.0,  # кг/кг
        hcl_yield=3.0  # кг/кг (для бензина нет)
    )

    # Параметры разлива
    liquid = LiquidFuelParams(
        density=750,  # кг/м³
        thickness=0.02,  # м (5 мм)
        spread_rate=0.0001  # м²/с
    )

    # Создаем калькулятор
    calc = FireHazardCalculator(room, gasoline, toxic, liquid)

    # Рассчитываем время блокирования
    results = calc.calculate_blocking_time()

    # Выводим результаты
    print("\nРезультаты расчета пожара пролива бензина:")
    print(f"Начальная площадь пролива: {room['burn_area']:.1f} м²")
    print(f"Скорость распространения: {liquid.spread_rate:.1f} м²/с")

    for factor, time in results.items():
        if time is not None:
            if factor == 'blocking_time':
                print(f"\nМинимальное время блокирования: {time:.1f} с")
            else:
                print(f"Время блокирования по {factor}: {time:.1f} с")
        else:
            print(f"Критическое значение по {factor} не достигнуто")

    # Выводим дополнительную информацию
    t_block = results['blocking_time']
    if t_block is not None:
        final_area = calc.calculate_burn_area(t_block)
        burned = calc.burned_mass(t_block)
        print(f"\nДополнительная информация на момент блокирования ({t_block:.1f} с):")
        print(f"Площадь пожара: {final_area:.1f} м²")
        print(f"Масса выгоревшего топлива: {burned:.1f} кг")
        print(f"Температура в помещении: {calc.temperature(t_block) - 273.15:.1f}°C")

        if calc.toxic_params:
            tox = calc.toxic_concentrations(t_block)
            print("\nКонцентрации токсичных веществ:")
            print(f"CO: {tox['CO']:.4f} кг/м³")
            print(f"CO2: {tox['CO2']:.4f} кг/м³")

    # Строим графики
    calc.plot_dynamics()

    # Сохраняем результаты
    calc.save_results(results, 'gasoline_fire_results.json')


if __name__ == "__main__":
    main()
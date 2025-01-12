import math
import dataclasses
from typing import List, Tuple, Optional
from enum import Enum


class ThermalDose(Enum):
    """Пороговые значения дозы теплового излучения"""
    SEVERE = 600  # Летальный исход с вероятностью близкой к 1
    HIGH = 320  # Тяжелые ожоги
    MEDIUM = 220  # Ожоги средней степени тяжести
    LOW = 120  # Легкие ожоги


@dataclasses.dataclass
class FireballParams:
    """Параметры огненного шара в конкретной точке"""
    distance: float  # Расстояние от центра, м
    intensity: float  # Интенсивность излучения, кВт/м2
    dose: float  # Доза излучения, кДж/м2
    probit: Optional[float]  # Пробит-функция
    probability: Optional[float]  # Вероятность поражения


class FireballCalculator:
    """
    Калькулятор параметров огненного шара по модели МЧС №404
    """

    def __init__(self, mass: float, surface_density: float):
        """
        Args:
            mass: Масса горючего вещества, кг
            surface_density: Средняя поверхностная плотность теплового излучения, кВт/м2
        """
        if mass <= 0 or surface_density <= 0:
            raise ValueError("Масса и плотность теплового излучения должны быть положительными")

        self.mass = mass
        self.surface_density = surface_density

        # Рассчитываем базовые параметры
        self.diameter = self._calc_diameter()
        self.height = self.diameter / 2
        self.lifetime = self._calc_lifetime()

    def _calc_diameter(self) -> float:
        """Расчет эффективного диаметра огненного шара"""
        return 5.33 * math.pow(self.mass, 0.327)

    def _calc_lifetime(self) -> float:
        """Расчет времени существования огненного шара"""
        return 0.92 * math.pow(self.mass, 0.303)

    def _calc_view_factor(self, distance: float) -> float:
        """
        Расчет углового коэффициента облученности

        Args:
            distance: Расстояние от центра огненного шара, м
        """
        h_ratio = self.height / self.diameter
        r_ratio = distance / self.diameter

        numerator = h_ratio + 0.5
        denominator = 4 * math.pow(math.pow(h_ratio + 0.5, 2) + math.pow(r_ratio, 2), 1.5)

        return numerator / denominator

    def _calc_atmosphere_transmission(self, distance: float) -> float:
        """
        Расчет коэффициента пропускания атмосферы

        Args:
            distance: Расстояние от центра огненного шара, м
        """
        path_length = math.sqrt(distance ** 2 + self.height ** 2) - self.diameter / 2
        return math.exp(-7e-4 * path_length)

    def _calc_probit(self, intensity: float) -> float:
        """
        Расчет пробит-функции для теплового излучения
        Значение пробит-функции не может быть отрицательным

        Args:
            intensity: Интенсивность теплового излучения, кВт/м2
        Returns:
            float: Значение пробит-функции (>= 0)
        """
        probit = -12.8 + 2.56 * math.log(intensity * self.lifetime)
        return max(0, probit)  # Ограничиваем минимальное значение нулем

    def _calc_probability(self, probit: float) -> float:
        """
        Расчет вероятности поражения по значению пробит-функции

        Args:
            probit: Значение пробит-функции
        """
        return 0.5 * (1 + math.erf((probit - 5) / math.sqrt(2)))

    def calculate_at_distance(self, distance: float) -> FireballParams:
        """
        Расчет параметров огненного шара на заданном расстоянии

        Args:
            distance: Расстояние от центра огненного шара, м

        Returns:
            FireballParams: Параметры огненного шара в заданной точке
        """
        if distance <= 0:
            raise ValueError("Расстояние должно быть положительным")

        view_factor = self._calc_view_factor(distance)
        transmission = self._calc_atmosphere_transmission(distance)

        intensity = self.surface_density * view_factor * transmission
        dose = intensity * self.lifetime

        probit = self._calc_probit(intensity)
        probability = self._calc_probability(probit)

        return FireballParams(
            distance=distance,
            intensity=round(intensity, 2),
            dose=round(dose, 2),
            probit=round(probit, 2),
            probability=round(probability, 4)
        )

    def find_hazard_zones(self) -> dict[ThermalDose, float]:
        """
        Определение радиусов зон поражения для различных пороговых значений дозы

        Returns:
            dict: Словарь с радиусами зон для каждого порогового значения
        """
        zones = {}

        # Начинаем с малого расстояния и увеличиваем его
        distance = 1.0
        results = []

        while True:
            params = self.calculate_at_distance(distance)
            if params.intensity < 1.2:  # Минимальный порог
                break

            results.append(params)
            distance += 0.5

        # Находим радиусы для каждой зоны
        for zone_type in ThermalDose:
            # Ищем ближайшее значение дозы к пороговому
            target_dose = zone_type.value
            closest = min(results, key=lambda x: abs(x.dose - target_dose))
            zones[zone_type] = closest.distance

        return zones

    def plot_parameters(self, max_distance: float = None) -> None:
        """
        Построение графиков зависимости параметров от расстояния

        Args:
            max_distance: Максимальное расстояние для построения графиков, м
                        Если не указано, рассчитывается автоматически
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Явно указываем бэкенд
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Для построения графиков необходимо установить matplotlib")

        # Если максимальное расстояние не задано, находим его автоматически
        if max_distance is None:
            distance = 1.0
            while self.calculate_at_distance(distance).intensity > 1.2:
                distance += 0.5
            max_distance = distance

        # Генерируем точки для построения
        distances = [d for d in range(1, int(max_distance) + 1)]
        params = [self.calculate_at_distance(d) for d in distances]

        # Извлекаем данные для графиков
        intensities = [p.intensity for p in params]
        doses = [p.dose for p in params]
        probits = [p.probit for p in params]
        probabilities = [p.probability for p in params]

        # Создаем сетку графиков 2x2
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Параметры огненного шара (масса {self.mass} кг)', fontsize=16)

        # График интенсивности излучения
        ax1.plot(distances, intensities, 'r-')
        ax1.set_title('Интенсивность излучения')
        ax1.set_xlabel('Расстояние, м')
        ax1.set_ylabel('Интенсивность, кВт/м²')
        ax1.grid(True)

        # График дозы излучения
        ax2.plot(distances, doses, 'b-')
        ax2.set_title('Доза излучения')
        ax2.set_xlabel('Расстояние, м')
        ax2.set_ylabel('Доза, кДж/м²')
        ax2.grid(True)

        # График пробит-функции
        ax3.plot(distances, probits, 'g-')
        ax3.set_title('Пробит-функция')
        ax3.set_xlabel('Расстояние, м')
        ax3.set_ylabel('Значение пробит-функции')
        ax3.grid(True)

        # График вероятности поражения
        ax4.plot(distances, probabilities, 'm-')
        ax4.set_title('Вероятность поражения')
        ax4.set_xlabel('Расстояние, м')
        ax4.set_ylabel('Вероятность')
        ax4.grid(True)

        # Добавляем линии зон поражения на график дозы
        zones = self.find_hazard_zones()
        colors = ['r', 'orange', 'y', 'g']
        zone_names = {
            ThermalDose.SEVERE: "Летальный исход",
            ThermalDose.HIGH: "Тяжелые ожоги",
            ThermalDose.MEDIUM: "Средние ожоги",
            ThermalDose.LOW: "Легкие ожоги"
        }
        for (zone_type, radius), color in zip(zones.items(), colors):
            ax2.axvline(x=radius, color=color, linestyle='--',
                        label=f'{zone_names[zone_type]} ({zone_type.value} кДж/м²)')
        ax2.legend()

        plt.tight_layout()

        # Сохраняем график в файл вместо показа
        plt.savefig('fireball_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self) -> str:
        """
        Генерация текстового отчета с результатами расчетов

        Returns:
            str: Форматированный отчет
        """
        zones = self.find_hazard_zones()

        report = [
            "Отчет по расчету параметров огненного шара",
            "=" * 50,
            f"Масса вещества: {self.mass:.2f} кг",
            f"Плотность излучения: {self.surface_density:.2f} кВт/м2",
            f"Диаметр огненного шара: {self.diameter:.2f} м",
            f"Время существования: {self.lifetime:.2f} с",
            "\nРадиусы зон поражения:",
            "-" * 30
        ]

        for zone_type, radius in zones.items():
            report.append(f"{zone_type.name}: {radius:.2f} м")

        return "\n".join(report)


def example_usage():
    """Пример использования калькулятора"""
    # Параметры из ГОСТ 12.3.047-98
    mass = 1760  # кг
    surface_density = 350  # кВт/м2

    calc = FireballCalculator(mass, surface_density)

    # Печать отчета
    print(calc.generate_report())

    # Расчет параметров на конкретном расстоянии
    params = calc.calculate_at_distance(100)
    print(f"\nПараметры на расстоянии {params.distance} м:")
    print(f"Интенсивность излучения: {params.intensity} кВт/м2")
    print(f"Доза излучения: {params.dose} кДж/м2")
    print(f"Вероятность поражения: {params.probability:.2%}")

    # Построение графиков
    calc.plot_parameters()


if __name__ == '__main__':
    example_usage()
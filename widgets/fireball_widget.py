# widgets/fireball_calculator.py
from PySide6.QtWidgets import QDoubleSpinBox
from widgets.base_calculator import BaseCalculatorWidget
from models.Fireball.fireball import FireballCalculator, ThermalDose  # Добавляем импорт ThermalDose
import matplotlib.pyplot as plt
import numpy as np


class FireballCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета параметров огненного шара"""

    def setup_inputs(self):
        """Настройка полей ввода для огненного шара"""
        # Масса вещества
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.1, 10000)
        self.mass_input.setValue(1760)
        self.mass_input.setDecimals(1)
        self.input_layout.addRow("Масса вещества [кг]:", self.mass_input)

        # Плотность излучения
        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(100, 500)
        self.density_input.setValue(350)
        self.density_input.setDecimals(1)
        self.input_layout.addRow("Плотность излучения [кВт/м²]:", self.density_input)

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {
            'mass': self.mass_input.value(),
            'surface_density': self.density_input.value()
        }

    def plot_results(self, calculator, max_distance=None):
        """Построение графиков на виджете"""
        # Очищаем предыдущие графики
        self.figure.clear()

        # Если максимальное расстояние не задано, находим его автоматически
        if max_distance is None:
            distance = 1.0
            while calculator.calculate_at_distance(distance).intensity > 1.2:
                distance += 0.5
            max_distance = distance

        # Генерируем точки для построения
        distances = [d for d in range(1, int(max_distance) + 1)]
        params = [calculator.calculate_at_distance(d) for d in distances]

        # Извлекаем данные для графиков
        intensities = [p.intensity for p in params]
        doses = [p.dose for p in params]
        probits = [p.probit for p in params]
        probabilities = [p.probability for p in params]

        # Создаем сетку графиков 2x2
        ((ax1, ax2), (ax3, ax4)) = self.figure.subplots(2, 2)
        self.figure.suptitle(f'Параметры огненного шара (масса {calculator.mass} кг)', fontsize=12)

        # График интенсивности излучения
        ax1.plot(distances, intensities, 'r-')
        ax1.set_title('Интенсивность излучения', fontsize=8)
        ax1.set_xlabel('Расстояние, м', fontsize=8)
        ax1.set_ylabel('Интенсивность, кВт/м²', fontsize=8)
        ax1.grid(True)

        # График дозы излучения
        ax2.plot(distances, doses, 'b-')
        ax2.set_title('Доза излучения', fontsize=8)
        ax2.set_xlabel('Расстояние, м', fontsize=8)
        ax2.set_ylabel('Доза, кДж/м²', fontsize=8)
        ax2.grid(True)

        # График пробит-функции
        ax3.plot(distances, probits, 'g-')
        ax3.set_title('Пробит-функция', fontsize=8)
        ax3.set_xlabel('Расстояние, м', fontsize=8)
        ax3.set_ylabel('Значение пробит-функции', fontsize=8)
        ax3.grid(True)

        # График вероятности поражения
        ax4.plot(distances, probabilities, 'm-')
        ax4.set_title('Вероятность поражения', fontsize=8)
        ax4.set_xlabel('Расстояние, м', fontsize=8)
        ax4.set_ylabel('Вероятность', fontsize=8)
        ax4.grid(True)

        # Добавляем линии зон поражения на график дозы
        zones = calculator.find_hazard_zones()
        colors = ['r', 'orange', 'y', 'g']
        zone_names = {
            ThermalDose.SEVERE: "Летальный исход",  # Используем импортированный ThermalDose
            ThermalDose.HIGH: "Тяжелые ожоги",
            ThermalDose.MEDIUM: "Средние ожоги",
            ThermalDose.LOW: "Легкие ожоги"
        }

        for (zone_type, radius), color in zip(zones.items(), colors):
            ax2.axvline(x=radius, color=color, linestyle='--',
                        label=f'{zone_names[zone_type]} ({zone_type.value} кДж/м²)')
        ax2.legend(fontsize=6)

        # Обновляем отображение
        self.figure.tight_layout()
        self.canvas.draw()

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Создание калькулятора и выполнение расчетов
            params = self.get_params()
            calculator = FireballCalculator(
                mass=params['mass'],
                surface_density=params['surface_density']
            )

            # Вывод отчета
            report = calculator.generate_report()
            self.output_text.setText(report)

            # Построение графиков непосредственно на виджете
            self.plot_results(calculator)

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
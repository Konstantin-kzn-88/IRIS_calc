# widgets/well_widget.py
from PySide6.QtWidgets import QDoubleSpinBox
from widgets.base_calculator import BaseCalculatorWidget
from models.Outflow_liguid_well.оutflow_well import calc_flow_rate_and_pressure
import numpy as np


class WellCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета истечения из скважины"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Создаем поля ввода
        self.inputs = {}

        # Геометрические параметры
        for param, (label, value, decimals) in {
            'L': ('Длина скважины [м]', 1000.0, 1),
            'd0': ('Диаметр на устье [м]', 0.2, 3),
            'd1': ('Диаметр на забое [м]', 0.1, 3),
            'theta': ('Угол наклона [градусы]', 30.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Физические параметры
        for param, (label, value, decimals) in {
            'P0': ('Давление в пласте [МПа]', 10.0, 2),
            'P_atm': ('Атмосферное давление [МПа]', 0.101325, 6),
            'rho': ('Плотность жидкости [кг/м³]', 1000.0, 1),
            'mu': ('Динамическая вязкость [Па·с]', 0.001, 6),
            'roughness': ('Шероховатость стенок [мм]', 0.01, 3),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            if param == 'mu':
                min_val = 1e-6  # минимальное значение для вязкости
                max_val = 1.0  # максимальное значение для вязкости
            elif param in ['P_atm', 'roughness']:
                min_val = 1e-6
                max_val = 100000
            else:
                min_val = 0.1
                max_val = 100000
            self.inputs[param].setRange(min_val, max_val)

            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        params = {key: widget.value() for key, widget in self.inputs.items()}

        # Преобразование единиц измерения
        params['P0'] *= 1e6  # МПа -> Па
        params['P_atm'] *= 1e6  # МПа -> Па
        params['roughness'] *= 0.001  # мм -> м
        params['theta'] = np.deg2rad(params['theta'])  # градусы -> радианы
        params['g'] = 9.81  # добавляем ускорение свободного падения

        return params

    def calculate_velocity_and_reynolds(self, Q, d, rho, mu):
        """Безопасный расчет скорости и числа Рейнольдса"""
        # Защита от деления на ноль при расчете скорости
        area = np.pi * d * d / 4
        velocity = np.zeros_like(Q)
        mask = area > 0
        velocity[mask] = Q[mask] / area[mask]

        # Безопасный расчет числа Рейнольдса
        Re = np.zeros_like(velocity)
        if mu > 0:
            Re = rho * np.abs(velocity) * d / mu

        return velocity, Re

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров
            params = self.get_params()

            # Проверка корректности входных данных
            if params['mu'] <= 0:
                raise ValueError("Вязкость должна быть положительной")
            if params['d0'] <= 0 or params['d1'] <= 0:
                raise ValueError("Диаметры должны быть положительными")

            # Выполнение расчета
            Q, P, x = calc_flow_rate_and_pressure(**params)

            # Расчет дополнительных параметров
            d = params['d0'] + (params['d1'] - params['d0']) * x / params['L']
            velocity, Re = self.calculate_velocity_and_reynolds(Q, d, params['rho'], params['mu'])

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Расход на устье: {Q[-1]:.3f} м³/с ({Q[-1] * 1000:.1f} л/с)")
            self.output_text.append(
                f"Скорость на устье: {velocity[-1]:.2f} м/с")
            self.output_text.append(
                f"Число Рейнольдса на устье: {Re[-1]:.0f}")
            self.output_text.append(
                f"Давление на забое: {P[0] / 1e6:.2f} МПа")
            self.output_text.append(
                f"Давление на устье: {P[-1] / 1e6:.2f} МПа")

            # Очистка и подготовка графиков
            self.figure.clear()

            # Создаем подграфики
            ((ax1, ax2), (ax3, ax4)) = self.figure.subplots(2, 2)

            # Преобразуем координату x в глубину для более наглядного отображения
            depth = params['L'] - x

            # График давления
            ax1.plot(P / 1e6, depth, 'b-', linewidth=2)
            ax1.set_xlabel('Давление [МПа]')
            ax1.set_ylabel('Глубина [м]')
            ax1.grid(True)
            ax1.set_title('Распределение давления')
            ax1.invert_yaxis()

            # График расхода
            ax2.plot(Q * 1000, depth, 'r-', linewidth=2)
            ax2.set_xlabel('Расход [л/с]')
            ax2.set_ylabel('Глубина [м]')
            ax2.grid(True)
            ax2.set_title('Распределение расхода')
            ax2.invert_yaxis()

            # График дебита
            ax3.plot(Q, depth, 'g-', linewidth=2)
            ax3.set_xlabel('Дебит [м³/с]')
            ax3.set_ylabel('Глубина [м]')
            ax3.grid(True)
            ax3.set_title('Распределение дебита')
            ax3.invert_yaxis()

            # График числа Рейнольдса
            ax4.plot(Re / 1e3, depth, 'm-', linewidth=2)
            ax4.set_xlabel('Число Рейнольдса [×10³]')
            ax4.set_ylabel('Глубина [м]')
            ax4.grid(True)
            ax4.set_title('Распределение числа Рейнольдса')
            ax4.invert_yaxis()

            # Добавление информации о параметрах
            info_text = (
                f"Параметры расчета:\n"
                f"L = {params['L']:.1f} м\n"
                f"d₀ = {params['d0']:.3f} м\n"
                f"d₁ = {params['d1']:.3f} м\n"
                f"θ = {np.rad2deg(params['theta']):.1f}°\n"
                f"P₀ = {params['P0'] / 1e6:.1f} МПа\n"
                f"ρ = {params['rho']:.1f} кг/м³\n"
                f"μ = {params['mu']:.4f} Па·с\n"
                f"k = {params['roughness'] * 1000:.2f} мм"
            )
            self.figure.text(0.02, 0.02, info_text, fontsize=8,
                             bbox=dict(boxstyle='round',
                                       facecolor='white',
                                       alpha=0.8))

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
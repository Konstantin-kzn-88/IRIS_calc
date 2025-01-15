from PySide6.QtWidgets import QDoubleSpinBox
from widgets.base_calculator import BaseCalculatorWidget
from models.Outflow_liguid_well.оutflow_well import (
    calc_flow_rate_and_pressure,
    plot_results,
    save_results
)
import numpy as np


class WellCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета истечения жидкости из скважины"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Создаем поля ввода
        self.inputs = {}

        # Параметры скважины
        for param, (label, value, decimals) in {
            'L': ('Длина скважины [м]', 1000.0, 1),
            'd0': ('Диаметр на устье [м]', 0.2, 3),
            'd1': ('Диаметр на забое [м]', 0.1, 3),
            'theta': ('Угол наклона [градусы]', 30.0, 1),
            'P0': ('Давление в пласте [МПа]', 10.0, 2),
            'P_atm': ('Атмосферное давление [МПа]', 0.101325, 6),
            'rho': ('Плотность жидкости [кг/м³]', 1000.0, 1),
            'mu': ('Динамическая вязкость [Па·с]', 0.001, 4),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            if param.startswith('P'):
                self.inputs[param].setRange(0, 100)
            elif param == 'theta':
                self.inputs[param].setRange(0, 90)
            else:
                self.inputs[param].setRange(0.0001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        params = {key: widget.value() for key, widget in self.inputs.items()}
        # Преобразование давления из МПа в Па
        params['P0'] *= 1e6
        params['P_atm'] *= 1e6
        # Преобразование угла из градусов в радианы
        params['theta'] = np.deg2rad(params['theta'])
        # Добавляем ускорение свободного падения
        params['g'] = 9.81
        return params

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров
            params = self.get_params()

            # Выполнение расчета
            Q, P, x = calc_flow_rate_and_pressure(**params)

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Интенсивность истечения: {Q[-1]:.2f} м³/с ({Q[-1]*1000:.1f} л/с)")
            self.output_text.append(
                f"Давление на устье: {P[0]/1e6:.2f} МПа")
            self.output_text.append(
                f"Давление на забое: {P[-1]/1e6:.2f} МПа")

            # Расчет диаметра для каждой точки
            d = params['d0'] + (params['d1'] - params['d0']) * x / params['L']

            # Построение графиков
            self.figure.clear()
            plot_results(P, x, Q, d, params, self.figure)
            self.figure.tight_layout()
            self.canvas.draw()

            # Сохранение результатов в файл
            save_results(x, P, Q)

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
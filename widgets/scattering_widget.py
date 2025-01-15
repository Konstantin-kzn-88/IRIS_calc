from PySide6.QtWidgets import QDoubleSpinBox, QCheckBox
from widgets.base_calculator import BaseCalculatorWidget
from models.Scattering_fragments import scattering


class ScatteringCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета разлета осколков при взрыве резервуара"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Создаем поля ввода
        self.inputs = {}

        # Параметры резервуара
        for param, (label, value, decimals) in {
            'P0': ('Избыточное давление [МПа]', 0.8, 2),
            'V0': ('Объем резервуара [м³]', 100.0, 1),
            'M_ob': ('Масса оболочки [кг]', 4000.0, 1),
            'rho_ob': ('Плотность материала оболочки [кг/м³]', 7850.0, 1),
            'n_fragments': ('Количество осколков [шт]', 5.0, 0),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.1, 1e6)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Чекбокс для выбора типа резервуара
        self.is_spherical = QCheckBox("Сферический резервуар")
        self.input_layout.addRow(self.is_spherical)

    def get_params(self):
        """Получение параметров из полей ввода"""
        params = {key: widget.value() for key, widget in self.inputs.items()}
        params['is_spherical'] = self.is_spherical.isChecked()
        return params

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Получение параметров
            params = self.get_params()

            # Выполнение расчета
            results = scattering.analyze_fragments(**params)

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append(
                f"Начальная скорость осколков: {results['initial_velocity']:.1f} м/с")
            self.output_text.append(
                f"Эффективная энергия взрыва: {results['effective_energy'] / 1e6:.1f} МДж")
            self.output_text.append(
                f"Масса осколка: {results['fragment_mass']:.1f} кг")
            self.output_text.append(
                f"Параметр W: {results['parameter_W']:.2f}")
            self.output_text.append(
                f"Максимальная дальность разлета: {results['max_distance']:.1f} м")

            # Построение графиков
            scattering.plot_results(results, params, self.figure)
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
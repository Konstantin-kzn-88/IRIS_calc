# widgets/pipeline_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Outflow_pipe_liguid.PipelineLeak import (
    LeakCalculator, PipeParameters, LeakParameters, FluidType
)


class PipelineCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета аварийного истечения из трубопровода"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор типа жидкости
        fluid_layout = QHBoxLayout()
        fluid_label = QLabel("Тип жидкости:")
        self.fluid_combo = QComboBox()
        for fluid in FluidType:
            self.fluid_combo.addItem(fluid.value["name"], fluid)
        fluid_layout.addWidget(fluid_label)
        fluid_layout.addWidget(self.fluid_combo)
        fluid_layout.addStretch()
        self.input_layout.addRow(fluid_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры трубопровода
        for param, (label, value, decimals) in {
            'diameter': ('Диаметр трубы [м]', 0.1, 3),
            'length': ('Длина трубы [м]', 1000.0, 1),
            'roughness': ('Шероховатость [м]', 0.0001, 4),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры утечки
        for param, (label, value, decimals) in {
            'pressure_start': ('Начальное давление [МПа]', 5.0, 2),
            'pressure_end': ('Конечное давление [МПа]', 0.101325, 6),
            'height_diff': ('Разница высот [м]', -2.0, 1),
            'hole_diameter': ('Диаметр отверстия [м]', 0.01, 3),
            'discharge_coef': ('Коэффициент расхода', 0.62, 2),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(-10000 if param == 'height_diff' else 0.000001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        pipe_params = {
            key: self.inputs[key].value()
            for key in ['diameter', 'length', 'roughness']
        }

        leak_params = {
            key: self.inputs[key].value()
            for key in ['pressure_start', 'pressure_end', 'height_diff',
                        'hole_diameter', 'discharge_coef']
        }

        return {
            'pipe': pipe_params,
            'leak': leak_params,
            'fluid_type': self.fluid_combo.currentData()
        }

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Получение параметров
            params = self.get_params()

            # Создание объектов параметров
            pipe = PipeParameters(**params['pipe'])
            leak = LeakParameters(**params['leak'])

            # Создание калькулятора и выполнение расчетов
            calculator = LeakCalculator(pipe, leak, params['fluid_type'])
            initial_state = calculator.calculate_initial_state()
            time_array, results = calculator.calculate_time_series()

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Скорость истечения: {initial_state['velocity']:.2f} м/с")
            self.output_text.append(
                f"Объемный расход: {initial_state['volume_flow_hour']:.2f} м³/ч")
            self.output_text.append(
                f"Массовый расход: {initial_state['mass_flow']:.2f} кг/с")
            self.output_text.append(
                f"Время опорожнения: {initial_state['emptying_time_min']:.1f} мин")
            self.output_text.append(
                f"Число Рейнольдса: {initial_state['reynolds']:.0f}")
            self.output_text.append(
                f"Коэффициент трения: {initial_state['friction_factor']:.4f}")
            self.output_text.append(
                f"Потери давления: {initial_state['pressure_drop_mpa']:.3f} МПа")
            self.output_text.append(
                f"Эффективное давление: {initial_state['effective_pressure_mpa']:.3f} МПа")

            # Построение графиков
            axes = self.figure.subplots(3, 1)

            # График массового расхода
            axes[0].plot(time_array / 60, results["mass_flow"], 'b-', linewidth=2)
            axes[0].set_xlabel('Время, мин')
            axes[0].set_ylabel('Массовый расход, кг/с')
            axes[0].grid(True)
            axes[0].set_title('Изменение массового расхода')

            # График давления
            axes[1].plot(time_array / 60, results["pressure"], 'r-', linewidth=2)
            axes[1].set_xlabel('Время, мин')
            axes[1].set_ylabel('Давление, МПа')
            axes[1].grid(True)
            axes[1].set_title('Изменение давления')

            # График энергетических характеристик
            axes[2].plot(time_array / 60, results["kinetic_energy"], 'g-',
                         label='Кинетическая', linewidth=2)
            axes[2].plot(time_array / 60, results["potential_energy"], 'm-',
                         label='Потенциальная', linewidth=2)
            axes[2].plot(time_array / 60, results["total_energy"], 'k--',
                         label='Полная', linewidth=2)
            axes[2].set_xlabel('Время, мин')
            axes[2].set_ylabel('Энергия, Дж/кг')
            axes[2].grid(True)
            axes[2].set_title('Энергетические характеристики')
            axes[2].legend()

            # Добавляем информацию о параметрах
            info_text = (
                f'Параметры расчета:\n'
                f'Тип жидкости: {params["fluid_type"].value["name"]}\n'
                f'Начальное давление: {params["leak"]["pressure_start"]:.1f} МПа\n'
                f'Длина трубопровода: {params["pipe"]["length"]:.0f} м\n'
                f'Диаметр трубы: {params["pipe"]["diameter"] * 1000:.0f} мм\n'
                f'Диаметр отверстия: {params["leak"]["hole_diameter"] * 1000:.0f} мм'
            )
            self.figure.text(0.02, 0.02, info_text, fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.8))

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
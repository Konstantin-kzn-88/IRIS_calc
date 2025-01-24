# widgets/gas_outflow_pipe_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Outflow_gas_pipe.outflow_gas import GasLeakModel
import numpy as np


class GasOutflowCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета истечения газа из трубопровода"""

    # Предустановки для разных газов
    GAS_PRESETS = {
        'methane': {
            'name': 'Метан',
            'k': 1.32,
            'R': 518.3,
            'M': 0.016
        },
        'propane': {
            'name': 'Пропан',
            'k': 1.13,
            'R': 188.5,
            'M': 0.044
        },
        'nitrogen': {
            'name': 'Азот',
            'k': 1.4,
            'R': 296.8,
            'M': 0.028
        }
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор типа газа
        gas_layout = QHBoxLayout()
        gas_label = QLabel("Тип газа:")
        self.gas_combo = QComboBox()
        self.gas_combo.addItem("Выберите тип газа...")
        for preset_id, preset_data in self.GAS_PRESETS.items():
            self.gas_combo.addItem(preset_data['name'], preset_id)
        self.gas_combo.currentIndexChanged.connect(self.on_gas_changed)
        gas_layout.addWidget(gas_label)
        gas_layout.addWidget(self.gas_combo)
        gas_layout.addStretch()
        self.input_layout.addRow(gas_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры газа
        for param, (label, value, decimals) in {
            'k': ('Показатель адиабаты', 1.32, 3),
            'R': ('Газовая постоянная [Дж/(кг·К)]', 518.3, 1),
            'T_C': ('Температура газа [°C]', 20.0, 1),
            'M': ('Молярная масса [кг/моль]', 0.016, 4),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 1000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.inputs[param].valueChanged.connect(self.on_value_changed)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры истечения
        for param, (label, value, decimals) in {
            'P1': ('Давление в трубе [МПа]', 1.0, 2),
            'D': ('Диаметр отверстия [мм]', 10.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 1000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.inputs[param].valueChanged.connect(self.on_value_changed)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_gas_preset(self, index):
        """Применение предустановки газа"""
        if index <= 0:
            return

        preset_id = self.gas_combo.currentData()
        preset = self.GAS_PRESETS[preset_id]

        # Обновляем значения в полях ввода
        for param in ['k', 'R', 'M']:
            if param in preset:
                self.inputs[param].setValue(preset[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def on_gas_changed(self, index):
        """Обработчик изменения типа газа"""
        self.apply_gas_preset(index)
        self.on_value_changed()

    def on_value_changed(self):
        """Обработчик изменения значений в полях ввода"""
        try:
            self.update_results_and_plots()
        except Exception:
            pass  # Игнорируем ошибки при неполном вводе данных

    def update_results_and_plots(self):
        """Обновление результатов и графиков"""
        # Очистка предыдущих результатов
        self.output_text.clear()
        self.figure.clear()

        # Получение параметров
        params = self.get_params()

        # Создание газовой модели
        gas_properties = {
            'k': params['k'],
            'R': params['R'],
            'T_C': params['T_C'],
            'M': params['M']
        }
        model = GasLeakModel(gas_properties)

        # Расчет параметров истечения
        pr_crit = model.critical_pressure_ratio()
        mass_flow = model.mass_flow_rate(params['P1'], params['D'])
        velocity = model.velocity(params['P1'])

        # Вывод результатов
        self.output_text.append("Результаты расчета:")
        self.output_text.append("-" * 50)
        self.output_text.append(f"Критическое отношение давлений: {pr_crit:.3f}")
        self.output_text.append(f"Массовый расход газа: {mass_flow:.4f} кг/с")
        # self.output_text.append(f"Объемный расход газа: {mass_flow * 3600 / gas_properties['M']:.1f} м³/ч")
        self.output_text.append(f"Скорость истечения: {velocity:.1f} м/с")

        # Построение графиков
        # Создаем сетку графиков 2x1
        (ax1, ax2) = self.figure.subplots(2, 1)

        # График зависимости от давления
        P1_range = np.linspace(0.11, 10.0, 100)
        mass_flows = [model.mass_flow_rate(P1, params['D']) for P1 in P1_range]
        velocities = [model.velocity(P1) for P1 in P1_range]

        # График массового расхода
        ax1.plot(P1_range, mass_flows, 'b-', linewidth=2)
        ax1.set_xlabel('Давление в трубе [МПа]')
        ax1.set_ylabel('Массовый расход [кг/с]')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Зависимость массового расхода от давления')

        # Добавляем точку текущего давления на график массового расхода
        ax1.plot(params['P1'], mass_flow, 'ro', label='Текущее значение')
        ax1.legend()

        # График скорости
        ax2.plot(P1_range, velocities, 'r-', linewidth=2)
        ax2.set_xlabel('Давление в трубе [МПа]')
        ax2.set_ylabel('Скорость истечения [м/с]')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Зависимость скорости истечения от давления')

        # Добавляем точку текущего давления на график скорости
        ax2.plot(params['P1'], velocity, 'bo', label='Текущее значение')
        ax2.legend()

        # Добавляем информацию о параметрах
        info_text = (
            f"Исходные параметры:\n"
            f"Тип газа: {self.gas_combo.currentText()}\n"
            f"Температура: {params['T_C']:.1f}°C\n"
            f"Давление: {params['P1']:.2f} МПа\n"
            f"Диаметр отверстия: {params['D']:.1f} мм"
        )
        ax1.text(0.02, 0.98, info_text,
                 transform=ax1.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          alpha=0.8),
                 fontsize=8)

        # Обновляем отображение
        self.figure.tight_layout()
        self.canvas.draw()

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            self.update_results_and_plots()
        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
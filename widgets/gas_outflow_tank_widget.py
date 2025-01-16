# widgets/gas_outflow_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Outflow_gas_tank.outflow_gas import GasFlowModel


class TankGasOutflowCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета истечения газа из резервуара"""

    # Предустановки для разных газов
    GAS_PRESETS = {
        'air': {
            'name': 'Воздух',
            'gamma': 1.4,
            'R': 287.05,
            'description': 'γ=1.4, R=287.05 Дж/(кг·К)'
        },
        'methane': {
            'name': 'Метан',
            'gamma': 1.32,
            'R': 518.3,
            'description': 'γ=1.32, R=518.3 Дж/(кг·К)'
        },
        'nitrogen': {
            'name': 'Азот',
            'gamma': 1.4,
            'R': 296.8,
            'description': 'γ=1.4, R=296.8 Дж/(кг·К)'
        },
        'hydrogen': {
            'name': 'Водород',
            'gamma': 1.41,
            'R': 4124.3,
            'description': 'γ=1.41, R=4124.3 Дж/(кг·К)'
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
            self.gas_combo.addItem(f"{preset_data['name']} ({preset_data['description']})", preset_id)
        self.gas_combo.currentIndexChanged.connect(self.apply_gas_preset)
        gas_layout.addWidget(gas_label)
        gas_layout.addWidget(self.gas_combo)
        gas_layout.addStretch()
        self.input_layout.addRow(gas_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры газа
        for param, (label, value, decimals) in {
            'gamma': ('Показатель адиабаты', 1.4, 3),
            'R': ('Газовая постоянная [Дж/(кг·К)]', 287.05, 2),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(1.0, 5000.0)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры резервуара и условий
        for param, (label, value, decimals) in {
            'V': ('Объем резервуара [м³]', 100.0, 1),
            'T0': ('Начальная температура [°C]', 20.0, 1),
            'P0': ('Начальное давление [МПа]', 1.0, 3),
            'D': ('Диаметр отверстия [мм]', 30.0, 1),
            'Pa': ('Атмосферное давление [МПа]', 0.101325, 6)
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(-50 if param == 'T0' else 0.000001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_gas_preset(self, index):
        """Применение предустановки газа"""
        if index <= 0:
            return

        preset_id = self.gas_combo.currentData()
        preset = self.GAS_PRESETS[preset_id]

        # Обновляем значения в полях ввода
        self.inputs['gamma'].setValue(preset['gamma'])
        self.inputs['R'].setValue(preset['R'])

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров и создание модели
            params = self.get_params()
            model = GasFlowModel(
                V=params['V'],
                T0_celsius=params['T0'],
                P0_MPa=params['P0'],
                gamma=params['gamma'],
                R=params['R'],
                D_mm=params['D'],
                Pa_MPa=params['Pa']
            )

            # Моделирование
            t_span = 800  # секунд
            dt = 0.1
            t, solution = model.simulate(t_span, dt)

            # Расчет массового расхода
            mdot = [model.mass_flow_rate(P, T) for P, T in zip(solution[:, 1], solution[:, 2])]

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(f"Начальная масса газа: {model.m0:.2f} кг")
            self.output_text.append(f"Начальный массовый расход: {mdot[0]:.3f} кг/с")
            self.output_text.append(f"Максимальный массовый расход: {max(mdot):.3f} кг/с")
            self.output_text.append(f"Конечная масса газа: {solution[-1, 0]:.2f} кг")
            self.output_text.append(f"Конечное давление: {solution[-1, 1] / 1e6:.3f} МПа")
            self.output_text.append(f"Конечная температура: {solution[-1, 2] - 273.15:.1f}°C")

            # Построение графиков
            self.figure.clear()

            # Создаем подграфики
            ((ax1, ax2), (ax3, ax4)) = self.figure.subplots(2, 2)

            # График давления
            ax1.plot(t, solution[:, 1] / 1e6, 'b-', linewidth=2)
            ax1.set_xlabel('Время [с]')
            ax1.set_ylabel('Давление [МПа]')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_title('Давление в резервуаре')

            # График массы
            ax2.plot(t, solution[:, 0], 'g-', linewidth=2)
            ax2.set_xlabel('Время [с]')
            ax2.set_ylabel('Масса [кг]')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_title('Масса газа в резервуаре')

            # График температуры
            ax3.plot(t, solution[:, 2] - 273.15, 'r-', linewidth=2)
            ax3.set_xlabel('Время [с]')
            ax3.set_ylabel('Температура [°C]')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_title('Температура газа')

            # График массового расхода
            ax4.plot(t, mdot, 'm-', linewidth=2)
            ax4.set_xlabel('Время [с]')
            ax4.set_ylabel('Расход [кг/с]')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_title('Массовый расход')

            # Добавляем информацию о параметрах
            gas_name = "Пользовательский"
            for preset_id, preset in self.GAS_PRESETS.items():
                if abs(preset['gamma'] - params['gamma']) < 0.01 and \
                   abs(preset['R'] - params['R']) < 0.1:
                    gas_name = preset['name']
                    break

            info_text = (
                f"Исходные параметры:\n"
                f"Газ: {gas_name}\n"
                f"γ = {params['gamma']:.3f}\n"
                f"R = {params['R']:.1f} Дж/(кг·К)\n"
                f"V = {params['V']:.1f} м³\n"
                f"P₀ = {params['P0']:.3f} МПа\n"
                f"T₀ = {params['T0']:.1f}°C\n"
                f"D = {params['D']:.1f} мм"
            )

            self.figure.text(0.02, 0.02, info_text,
                           fontsize=8,
                           bbox=dict(facecolor='white', alpha=0.8))

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
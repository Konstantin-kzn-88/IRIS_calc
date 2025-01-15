# widgets/tank_outflow_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Outflow_liguid_from_tank.outflow_tank import calculate_flow


class TankOutflowCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета истечения жидкости из резервуара"""

    # Предустановки для разных жидкостей
    PRESETS = {
        'water': {
            'name': 'Вода',
            'density': 1000,
        },
        'gasoline': {
            'name': 'Бензин',
            'density': 750,
        },
        'diesel': {
            'name': 'Дизельное топливо',
            'density': 850,
        },
        'oil': {
            'name': 'Нефть',
            'density': 870,
        }
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выпадающий список для выбора жидкости
        fluid_layout = QHBoxLayout()
        fluid_label = QLabel("Тип жидкости:")
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItem("Выберите тип жидкости...")
        for preset_id, preset_data in self.PRESETS.items():
            self.fluid_combo.addItem(preset_data['name'], preset_id)
        self.fluid_combo.currentIndexChanged.connect(self.apply_preset)
        fluid_layout.addWidget(fluid_label)
        fluid_layout.addWidget(self.fluid_combo)
        fluid_layout.addStretch()
        self.input_layout.addRow(fluid_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры резервуара и отверстия
        for param, (label, value, decimals) in {
            'h0': ('Начальная высота жидкости [м]', 10.0, 2),
            'd_hole_mm': ('Диаметр отверстия [мм]', 112.8, 1),
            'D_tank': ('Диаметр резервуара [м]', 2.0, 2),
            'P_gauge_initial_MPa': ('Начальное избыточное давление [МПа]', 0.2, 3),
            'h_gas_initial': ('Начальная высота газовой подушки [м]', 1.0, 2),
            'density': ('Плотность жидкости [кг/м³]', 1000, 0),
            't_max': ('Время моделирования [с]', 30.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_preset(self, index):
        """Применение предустановки"""
        if index <= 0:
            return

        preset_id = self.fluid_combo.currentData()
        preset = self.PRESETS[preset_id]

        # Обновляем значения в полях ввода
        self.inputs['density'].setValue(preset['density'])

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров
            params = self.get_params()

            # Выполнение расчета
            t, h, P_gauge, Q_mass, m_flowed = calculate_flow(
                h0=params['h0'],
                d_hole_mm=params['d_hole_mm'],
                D_tank=params['D_tank'],
                P_gauge_initial_MPa=params['P_gauge_initial_MPa'],
                h_gas_initial=params['h_gas_initial'],
                rho=params['density'],
                t_max=params['t_max']
            )

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Начальная высота жидкости: {params['h0']:.2f} м")
            self.output_text.append(
                f"Диаметр отверстия: {params['d_hole_mm']:.1f} мм")
            self.output_text.append(
                f"Диаметр резервуара: {params['D_tank']:.2f} м")
            self.output_text.append(
                f"Начальное давление: {params['P_gauge_initial_MPa']:.3f} МПа")

            # Расчет конечных значений
            final_height = h[-1]
            final_pressure = P_gauge[-1] / 1e6  # Переводим в МПа
            max_flow_rate = max(Q_mass)
            total_mass = m_flowed[-1]

            self.output_text.append("\nИтоговые значения:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Конечная высота жидкости: {final_height:.2f} м")
            self.output_text.append(
                f"Конечное давление: {final_pressure:.3f} МПа")
            self.output_text.append(
                f"Максимальный расход: {max_flow_rate:.2f} кг/с")
            self.output_text.append(
                f"Общая масса вытекшей жидкости: {total_mass:.2f} кг")

            # Выполнение расчета для построения графиков
            t, h, P_gauge, Q_mass, m_flowed = calculate_flow(
                h0=params['h0'],
                d_hole_mm=params['d_hole_mm'],
                D_tank=params['D_tank'],
                P_gauge_initial_MPa=params['P_gauge_initial_MPa'],
                h_gas_initial=params['h_gas_initial'],
                rho=params['density'],
                t_max=params['t_max']
            )

            # Очистка фигуры
            self.figure.clear()

            # Создаем подграфики
            gs = self.figure.add_gridspec(2, 2)
            ax1 = self.figure.add_subplot(gs[0, 0])
            ax2 = self.figure.add_subplot(gs[0, 1])
            ax3 = self.figure.add_subplot(gs[1, 0])
            ax4 = self.figure.add_subplot(gs[1, 1])

            # График высоты жидкости
            ax1.plot(t, h, 'b-', linewidth=2, label='Высота')
            ax1.set_xlabel('Время, с')
            ax1.set_ylabel('Высота жидкости, м')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_title('Изменение высоты жидкости')
            ax1.legend()

            # График давления
            ax2.plot(t, P_gauge / 1e6, 'r-', linewidth=2, label='Давление')
            ax2.set_xlabel('Время, с')
            ax2.set_ylabel('Избыточное давление, МПа')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_title('Изменение давления газовой подушки')
            ax2.legend()

            # График массового расхода
            ax3.plot(t, Q_mass, 'g-', linewidth=2, label='Расход')
            ax3.set_xlabel('Время, с')
            ax3.set_ylabel('Массовый расход, кг/с')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_title('Изменение массового расхода')
            ax3.legend()

            # График массы вытекшей жидкости
            ax4.plot(t, m_flowed, 'm-', linewidth=2, label='Масса вытекшей')
            ax4.set_xlabel('Время, с')
            ax4.set_ylabel('Масса, кг')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_title('Масса вытекшей жидкости')
            ax4.legend()

            # Добавляем информацию о параметрах
            info_text = (
                f"Исходные параметры:\n\n"
                f"Диаметр отверстия: {params['d_hole_mm']:.1f} мм\n"
                f"Диаметр резервуара: {params['D_tank']:.2f} м\n"
                f"Начальная высота: {params['h0']:.2f} м\n"
                f"Начальное давление: {params['P_gauge_initial_MPa']:.3f} МПа\n"
                f"Высота газовой подушки: {params['h_gas_initial']:.2f} м\n"
                f"Плотность жидкости: {params['density']:.0f} кг/м³"
            )
            self.figure.text(0.02, 0.02, info_text,
                             bbox=dict(facecolor='white', alpha=0.8),
                             fontsize=8)

            # Настраиваем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
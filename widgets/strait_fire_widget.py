# widgets/strait_fire_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Strait_fire import strait_fire


class StraitFireCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета пожара пролива"""

    # Предустановки для разных жидкостей
    PRESETS = {
        'gasoline': {
            'name': 'Бензин',
            'mol_mass': 95.3,
            't_boiling': 100,
            'm_sg': 0.06,
        },
        'kerosene': {
            'name': 'Керосин',
            'mol_mass': 144.0,
            't_boiling': 150,
            'm_sg': 0.04,
        },
        'diesel': {
            'name': 'Дизельное топливо',
            'mol_mass': 203.6,
            't_boiling': 280,
            'm_sg': 0.04,
        }
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выпадающий список для выбора предустановок
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Тип жидкости:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Выберите тип жидкости...")
        for preset_id, preset_data in self.PRESETS.items():
            self.preset_combo.addItem(preset_data['name'], preset_id)
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        self.input_layout.addRow(preset_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры жидкости
        for param, (label, value, decimals) in {
            'mol_mass': ('Молекулярная масса [кг/кмоль]', 95.3, 1),
            't_boiling': ('Температура кипения [°C]', 100, 1),
            'm_sg': ('Удельная массовая скорость выгорания [кг/(с·м²)]', 0.06, 3),
            'S_spill': ('Площадь пролива [м²]', 100, 1),
            'wind_velocity': ('Скорость ветра [м/с]', 1, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 50000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_preset(self, index):
        """Применение предустановки"""
        if index <= 0:
            return

        preset_id = self.preset_combo.currentData()
        preset = self.PRESETS[preset_id]

        # Обновляем значения в полях ввода
        for param in ['mol_mass', 't_boiling', 'm_sg']:
            if param in preset:
                self.inputs[param].setValue(preset[param])

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

            # Создание модели и выполнение расчета
            input_params = strait_fire.InputParameters(
                S_spill=params['S_spill'],
                m_sg=params['m_sg'],
                mol_mass=params['mol_mass'],
                t_boiling=params['t_boiling'],
                wind_velocity=params['wind_velocity']
            )

            model = strait_fire.StraitFire()

            # Получаем массивы данных для построения графиков
            radius_arr, q_term_arr, probit_arr, probability_arr = model.termal_radiation_array(input_params)

            # Расчет зон поражения
            zones = model.termal_class_zone(input_params)

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)

            # Расчет эффективного диаметра
            D_eff = model.calculate_effective_diameter(params['S_spill'])
            self.output_text.append(f"Эффективный диаметр пролива: {D_eff:.2f} м")

            # Вывод радиусов зон поражения
            self.output_text.append("\nРадиусы зон поражения:")
            for zone_value, radius in zip(model.CLASSIFIED_ZONES, zones):
                self.output_text.append(f"Зона {zone_value:4.1f} кВт/м² - радиус {radius:.2f} м")

            # Очищаем фигуру
            self.figure.clear()

            # Создаем подграфики
            (ax1, ax2, ax3) = self.figure.subplots(3, 1)

            # График интенсивности теплового излучения
            ax1.plot(radius_arr, q_term_arr, 'b-', linewidth=2)
            ax1.set_xlabel('Расстояние, м')
            ax1.set_ylabel('Интенсивность, кВт/м²')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_title('Интенсивность теплового излучения')

            # Добавляем исходные данные
            info_text = (
                f"Исходные параметры:\n\n"
                f"Площадь пролива: {params['S_spill']} м²\n"
                f"Скорость выгорания: {params['m_sg']} кг/(с·м²)\n"
                f"Молекулярная масса: {params['mol_mass']} кг/кмоль\n"
                f"Температура кипения: {params['t_boiling']} °C\n"
                f"Скорость ветра: {params['wind_velocity']} м/с"
            )
            ax1.text(0.02, 0.98, info_text,
                     transform=ax1.transAxes,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='white',
                               alpha=0.8),
                     fontsize=8)

            # График вероятности поражения
            ax2.plot(radius_arr, probability_arr, 'r-', linewidth=2)
            ax2.set_xlabel('Расстояние, м')
            ax2.set_ylabel('Вероятность поражения')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_title('Вероятность поражения')

            # График пробит-функции
            ax3.plot(radius_arr, probit_arr, 'g-', linewidth=2)
            ax3.set_xlabel('Расстояние, м')
            ax3.set_ylabel('Пробит-функция')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_title('Пробит-функция')

            # Добавляем подписи зон поражения на график интенсивности
            zone_colors = ['#FFE5E5', '#FFE5CC', '#FFFFCC', '#E5FFE5']
            prev_x = 0

            for i, zone_value in enumerate(model.CLASSIFIED_ZONES):
                idx = q_term_arr.index(strait_fire.get_nearest_value(q_term_arr, zone_value))
                x = radius_arr[idx]

                # Закрашиваем зону
                ax1.axvspan(prev_x, x, alpha=0.3, color=zone_colors[i])

                # Добавляем вертикальную линию и подпись
                ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
                ax1.text(x, max(q_term_arr) * 0.7, f'{zone_value} кВт/м²',
                         rotation=90, verticalalignment='bottom')
                prev_x = x

            # Настраиваем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")


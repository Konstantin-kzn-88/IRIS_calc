# widgets/explosion_widget.py
import numpy as np
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout, QCheckBox
from widgets.base_calculator import BaseCalculatorWidget
from models.Explosion_TVS import explosion


class ExplosionCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета параметров взрыва топливно-воздушной смеси"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор класса чувствительности
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Класс чувствительности:")
        self.sensitivity_combo = QComboBox()
        for sensitivity_class in explosion.SensitivityClass:
            self.sensitivity_combo.addItem(f"Класс {sensitivity_class.value}", sensitivity_class)
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_combo)
        sensitivity_layout.addStretch()
        self.input_layout.addRow(sensitivity_layout)

        sensitivity_tooltip = ("Классы чувствительности веществ:\n"
                               "1 - особо чувствительные вещества (ацетилен, пропадиен)\n"
                               "2 - чувствительные вещества (этилен, бутан, пропан)\n"
                               "3 - средне чувствительные вещества (метан, метанол)\n"
                               "4 - слабо чувствительные вещества (аммиак)")
        self.sensitivity_combo.setToolTip(sensitivity_tooltip)

        # Выбор класса загроможденности пространства
        space_layout = QHBoxLayout()
        space_label = QLabel("Класс загроможденности:")
        self.space_combo = QComboBox()
        for space_class in explosion.SpaceClass:
            self.space_combo.addItem(f"Класс {space_class.value}", space_class)
        space_layout.addWidget(space_label)
        space_layout.addWidget(self.space_combo)
        space_layout.addStretch()
        self.input_layout.addRow(space_layout)

        space_tooltip = ("Классы загроможденности пространства:\n"
                         "1 - сильно загроможденное (>30% преград)\n"
                         "2 - средне загроможденное (20-30% преград)\n"
                         "3 - слабо загроможденное (10-20% преград)\n"
                         "4 - свободное пространство (<10% преград)")
        self.space_combo.setToolTip(space_tooltip)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры вещества и условий
        for param, (label, value, decimals) in {
            'M_g': ('Масса горючего вещества [кг]', 100.0, 1),
            'q_g': ('Теплота сгорания [кДж/кг]', 46000.0, 1),
            'beta': ('Коэффициент использования запаса [0-1]', 1.0, 2),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            if param == 'M_g':
                self.inputs[param].setRange(0.001, 1e6)
                self.inputs[param].setToolTip("Масса горючего вещества, участвующего в создании\n"
                                              "топливно-воздушной смеси.")
            elif param == 'q_g':
                self.inputs[param].setRange(0.001, 1e6)
                self.inputs[param].setToolTip("Теплота сгорания горючего вещества.\n\n"
                                              "Типичные значения:\n"
                                              "• Водород: 120000 кДж/кг\n"
                                              "• Метан: 50000 кДж/кг\n"
                                              "• Пропан: 46000 кДж/кг\n"
                                              "• Бутан: 45000 кДж/кг\n"
                                              "• Ацетилен: 48000 кДж/кг")
            elif param == 'beta':
                self.inputs[param].setRange(0.001, 1.0)
                tooltip = ("Коэффициент использования запаса топлива:\n"
                           "• 1.0 - для газов (полное участие)\n"
                           "• 0.1-0.3 - для легкоиспаряющихся жидкостей\n"
                           "• 0.02-0.1 - для тяжелых углеводородов\n\n"
                           "Учитывает долю топлива, реально участвующую во взрыве,\n"
                           "с учетом неполного испарения, рассеивания и\n"
                           "неравномерности смешения с воздухом.")
                self.inputs[param].setToolTip(tooltip)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Чекбоксы для дополнительных параметров
        self.is_gas = QCheckBox("Газовая смесь")
        self.is_gas.setChecked(True)
        self.is_gas.setToolTip("Отметьте для газовой смеси (σ = 7)\n"
                               "Снимите для гетерогенной смеси (σ = 4)")
        self.input_layout.addRow(self.is_gas)

        self.is_ground_level = QCheckBox("Облако на поверхности земли")
        self.is_ground_level.setChecked(True)
        self.is_ground_level.setToolTip("Отметьте если облако расположено на поверхности земли\n"
                                        "(эффективная энергия удваивается)\n"
                                        "Снимите если облако находится в объеме")
        self.input_layout.addRow(self.is_ground_level)

    def get_params(self):
        """Получение параметров из полей ввода"""
        params = {key: widget.value() for key, widget in self.inputs.items()}
        params.update({
            'sensitivity_class': self.sensitivity_combo.currentData(),
            'space_class': self.space_combo.currentData(),
            'is_gas': self.is_gas.isChecked(),
            'is_ground_level': self.is_ground_level.isChecked()
        })
        return params

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров
            params = self.get_params()

            # Создание модели и выполнение расчета
            tvs = explosion.TVSExplosion(
                M_g=params['M_g'],
                q_g=params['q_g'],
                sensitivity_class=params['sensitivity_class'],
                space_class=params['space_class'],
                beta=params['beta'],
                is_gas=params['is_gas'],
                is_ground_level=params['is_ground_level']
            )

            # Расчет характерных значений давления
            self.output_text.append("Характерные значения давления и соответствующие расстояния:")
            self.output_text.append("-" * 50)

            for pressure in explosion.PRESSURES_OF_INTEREST:
                distance = tvs.find_distance_for_pressure(
                    pressure, params['M_g'], params['q_g'],
                    params['sensitivity_class'], params['space_class'],
                    None, None, params['beta'],
                    params['is_gas'], params['is_ground_level']
                )
                if distance is not None:
                    self.output_text.append(
                        f"Давление {pressure} кПа достигается на расстоянии {distance:.2f} м")
                else:
                    self.output_text.append(
                        f"Давление {pressure} кПа не достигается")

            # Построение графиков
            self.figure.clear()

            # График давления (линейный масштаб)
            ax1 = self.figure.add_subplot(221)
            ax2 = self.figure.add_subplot(222)
            ax3 = self.figure.add_subplot(223)
            ax4 = self.figure.add_subplot(224)

            # Расчет данных для графиков
            r_min, r_max = 0.1, 200
            num_points = 1000
            r_linear = np.linspace(r_min, r_max, num_points)
            r_log = np.logspace(np.log10(r_min), np.log10(r_max), num_points)

            # Расчет значений для графиков
            pressures_linear = []
            pressures_log = []
            probits = []
            probabilities = []

            for r in r_linear:
                results = tvs.calculate_explosion_parameters_base(r, params['M_g'], params['q_g'],
                                                                  params['sensitivity_class'], params['space_class'],
                                                                  None, None, params['beta'],
                                                                  params['is_gas'], params['is_ground_level'])
                pressures_linear.append(results['избыточное_давление [Па]'] / 1000)

            for r in r_log:
                results = tvs.calculate_explosion_parameters_base(r, params['M_g'], params['q_g'],
                                                                  params['sensitivity_class'], params['space_class'],
                                                                  None, None, params['beta'],
                                                                  params['is_gas'], params['is_ground_level'])
                pressures_log.append(results['избыточное_давление [Па]'] / 1000)
                probits.append(results['пробит_функция [-]'])
                probabilities.append(results['вероятность_поражения [-]'] * 100)

            # График давления (линейный масштаб)
            ax1.plot(r_linear, pressures_linear, 'b-')
            ax1.set_xlabel('Расстояние (м)')
            ax1.set_ylabel('Избыточное давление (кПа)')
            ax1.set_title('Давление (линейный масштаб)')
            ax1.grid(True)

            # График давления (логарифмический масштаб)
            ax2.semilogx(r_log, pressures_log, 'b-')
            ax2.set_xlabel('Расстояние (м)')
            ax2.set_ylabel('Избыточное давление (кПа)')
            ax2.set_title('Давление (логарифмический масштаб)')
            ax2.grid(True)

            # График пробит-функции
            ax3.semilogx(r_log, probits, 'r-')
            ax3.set_xlabel('Расстояние (м)')
            ax3.set_ylabel('Пробит-функция')
            ax3.set_title('Пробит-функция')
            ax3.grid(True)

            # График вероятности поражения
            ax4.semilogx(r_log, probabilities, 'g-')
            ax4.set_xlabel('Расстояние (м)')
            ax4.set_ylabel('Вероятность поражения (%)')
            ax4.set_title('Вероятность поражения')
            ax4.grid(True)

            # Добавление информации о параметрах
            info_text = (
                f"Параметры расчета:\n"
                f"Масса: {params['M_g']:.1f} кг\n"
                f"Теплота сгорания: {params['q_g']:.0f} кДж/кг\n"
                f"Класс чувствительности: {params['sensitivity_class'].value}\n"
                f"Класс загроможденности: {params['space_class'].value}\n"
                f"{'Газовая' if params['is_gas'] else 'Гетерогенная'} смесь\n"
                f"{'На поверхности' if params['is_ground_level'] else 'В объеме'}"
            )
            self.figure.text(0.02, 0.02, info_text, fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.8))

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
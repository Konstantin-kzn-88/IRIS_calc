# widgets/explosion_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Explosion_SP import explosion


class ExplosionCalculatorWidgetSP(BaseCalculatorWidget):
    """Виджет для расчета параметров взрыва газопаровоздушной смеси"""

    # Предустановки для разных веществ
    PRESETS = {
        'methane': {
            'name': 'Метан',
            'Q_combustion': 50240,
        },
        'propane': {
            'name': 'Пропан',
            'Q_combustion': 46360,
        },
        'hydrogen': {
            'name': 'Водород',
            'Q_combustion': 120910,
        },
        'acetylene': {
            'name': 'Ацетилен',
            'Q_combustion': 48225,
        },
        'oil': {
            'name': 'Нефть',
            'Q_combustion': 43590,
        },
        'gasoline': {
            'name': 'Бензин',
            'Q_combustion': 44000,
        }
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор типа вещества
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Тип вещества:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Выберите тип вещества...")
        for preset_id, preset_data in self.PRESETS.items():
            self.preset_combo.addItem(preset_data['name'], preset_id)
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        self.input_layout.addRow(preset_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Свойства вещества
        self.inputs['Q_combustion'] = QDoubleSpinBox()
        self.inputs['Q_combustion'].setRange(0, 1000000)
        self.inputs['Q_combustion'].setValue(50240)
        self.inputs['Q_combustion'].setDecimals(0)
        self.input_layout.addRow("Теплота сгорания [кДж/кг]:", self.inputs['Q_combustion'])

        # Параметры взрыва
        for param, (label, value, decimals) in {
            'mass': ('Масса вещества [кг]', 100.0, 1),
            'distance': ('Расстояние [м]', 20.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.1, 10000)
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
        self.inputs['Q_combustion'].setValue(preset['Q_combustion'])

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {
            'mass': self.inputs['mass'].value(),
            'Q_combustion': self.inputs['Q_combustion'].value(),
            'distance': self.inputs['distance'].value()
        }

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров
            params = self.get_params()

            # Получение названия вещества
            substance_name = "Пользовательское вещество"
            if self.preset_combo.currentIndex() > 0:
                substance_name = self.preset_combo.currentText()

            # Выполнение расчета
            m_pr = explosion.calculate_equivalent_mass(
                params['mass'],
                params['Q_combustion']
            )

            # Расчет параметров на заданном расстоянии
            pressure = explosion.calculate_excess_pressure(m_pr, params['distance'])
            impulse = explosion.calculate_pressure_impulse(m_pr, params['distance'])
            probit, probability = explosion.calculate_damage_probability(pressure, impulse)

            # Расчет расстояний для характерных давлений
            distances = {}
            for p in explosion.PRESSURES_OF_INTEREST:
                dist = explosion.find_distance_for_pressure(m_pr, p)
                imp = explosion.calculate_pressure_impulse(m_pr, dist)
                pr, prob = explosion.calculate_damage_probability(p, imp)
                distances[p] = {
                    'distance': dist,
                    'impulse': imp,
                    'probit': pr,
                    'probability': prob
                }

            # Формируем результаты в том же формате, что ожидает метод вывода
            results = {
                'equivalent_mass': m_pr,
                'excess_pressure': pressure,
                'pressure_impulse': impulse,
                'probit': probit,
                'probability': probability,
                'distances_for_pressures': distances
            }

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Коэффициент участия: {explosion.Z_COEFFICIENT}")
            self.output_text.append(
                f"Приведенная масса: {results['equivalent_mass']:.2f} кг")
            self.output_text.append(
                f"Избыточное давление: {results['excess_pressure']:.2f} кПа")
            self.output_text.append(
                f"Импульс волны давления: {results['pressure_impulse']:.2f} Па·с")
            self.output_text.append(
                f"Пробит-функция: {results['probit']:.2f}")
            self.output_text.append(
                f"Вероятность поражения: {results['probability']:.2%}")

            self.output_text.append("\nРасстояния для характерных давлений:")
            for pressure, data in results['distances_for_pressures'].items():
                self.output_text.append(f"\nДавление {pressure} кПа:")
                self.output_text.append(f"  Расстояние: {data['distance']:.2f} м")
                self.output_text.append(f"  Импульс: {data['impulse']:.2f} Па·с")
                self.output_text.append(f"  Вероятность поражения: {data['probability']:.2%}")

            # Построение графиков
            self.figure.clear()

            # Создаем массив расстояний для построения графиков
            import numpy as np
            distances = np.logspace(np.log10(0.1), np.log10(200), 300)

            # Расчет значений для графиков
            pressures = [explosion.calculate_excess_pressure(m_pr, r) for r in distances]
            impulses = [explosion.calculate_pressure_impulse(m_pr, r) for r in distances]
            probits = [explosion.calculate_probit(p, i) for p, i in zip(pressures, impulses)]
            probabilities = [explosion.calculate_probability(pr) * 100 for pr in probits]

            # Создаем подграфики
            ((ax1, ax2), (ax3, ax4)) = self.figure.subplots(2, 2)

            # График давления
            ax1.semilogx(distances, pressures, 'b-', linewidth=2)
            ax1.set_xlabel('Расстояние, м')
            ax1.set_ylabel('Избыточное давление, кПа')
            ax1.grid(True, which='both', linestyle='--', alpha=0.7)
            ax1.set_title('Зависимость давления от расстояния')

            # График импульса
            ax2.semilogx(distances, impulses, 'g-', linewidth=2)
            ax2.set_xlabel('Расстояние, м')
            ax2.set_ylabel('Импульс, Па·с')
            ax2.grid(True, which='both', linestyle='--', alpha=0.7)
            ax2.set_title('Зависимость импульса от расстояния')

            # График пробит-функции
            ax3.semilogx(distances, probits, 'r-', linewidth=2)
            ax3.set_xlabel('Расстояние, м')
            ax3.set_ylabel('Пробит-функция')
            ax3.grid(True, which='both', linestyle='--', alpha=0.7)
            ax3.set_title('Зависимость пробит-функции от расстояния')

            # График вероятности поражения
            ax4.semilogx(distances, probabilities, 'y-', linewidth=2)
            ax4.set_xlabel('Расстояние, м')
            ax4.set_ylabel('Вероятность поражения, %')
            ax4.grid(True, which='both', linestyle='--', alpha=0.7)
            ax4.set_title('Зависимость вероятности поражения от расстояния')

            # Добавляем информацию о параметрах
            info_text = (
                f"Исходные параметры:\n"
                f"Вещество: {substance_name}\n"
                f"Масса: {params['mass']:.1f} кг\n"
                f"Теплота сгорания: {params['Q_combustion']:.0f} кДж/кг\n"
                f"Расстояние: {params['distance']:.1f} м\n"
                f"Приведенная масса: {results['equivalent_mass']:.1f} кг"
            )

            self.figure.text(0.02, 0.02, info_text,
                           bbox=dict(facecolor='white', alpha=0.8),
                           fontsize=8)

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
# widgets/jet_fire_widget.py
import math

from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.Jet_fire.calc_jet_fire import Torch
import numpy as np


class JetFireCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета факельного горения"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор типа вещества
        substance_layout = QHBoxLayout()
        substance_label = QLabel("Тип вещества:")
        self.substance_combo = QComboBox()
        self.substance_combo.addItems([
            "Газ",
            "Газ СУГ",
            "СУГ"
        ])
        substance_layout.addWidget(substance_label)
        substance_layout.addWidget(self.substance_combo)
        substance_layout.addStretch()
        self.input_layout.addRow(substance_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры расчета
        self.inputs['consumption'] = QDoubleSpinBox()
        self.inputs['consumption'].setRange(0.001, 1000)
        self.inputs['consumption'].setValue(1.0)
        self.inputs['consumption'].setDecimals(3)
        self.input_layout.addRow("Расход вещества [кг/с]:", self.inputs['consumption'])

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {
            'consumption': self.inputs['consumption'].value(),
            'type': self.substance_combo.currentIndex()
        }

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Получение параметров
            params = self.get_params()

            # Создание модели и выполнение расчета
            model = Torch()

            # Создаем сетку графиков
            axes = self.figure.subplots(2, 1)

            # Генерируем массив расходов для построения графиков
            consumptions = np.linspace(0.001, max(params['consumption'] * 2, 10), 1000)

            # Рассчитываем теоретические значения
            TYPE_COEF = (12.5, 13.5, 15)  # коэффициенты для разных типов
            coef = TYPE_COEF[params['type']]

            # Теоретические кривые
            lengths = coef * np.power(consumptions, 0.4)
            diameters = 0.15 * lengths

            # Текущие значения без округления (для точки на графике)
            Lf_current = coef * math.pow(params['consumption'], 0.4)
            Df_current = 0.15 * Lf_current

            # Округленные значения для вывода в текст
            Lf_rounded = int(Lf_current)
            Df_rounded = math.ceil(Df_current)

            # Вывод результатов в текстовое поле
            self.output_text.append("Результаты расчета:")
            self.output_text.append(f"Длина факела: {Lf_rounded} м")
            self.output_text.append(f"Диаметр факела: {Df_rounded} м")

            # Настройка стиля графиков
            for ax in axes:
                ax.grid(True, which='both', linestyle='--', alpha=0.7)
                ax.minorticks_on()
                ax.grid(True, which='minor', linestyle=':', alpha=0.4)

            # График зависимости длины факела от расхода
            axes[0].plot(consumptions, lengths, 'r-', linewidth=2)
            axes[0].plot(params['consumption'], Lf_current, 'bo', label='Текущее значение')
            axes[0].set_xlabel('Расход, кг/с')
            axes[0].set_ylabel('Длина факела, м')
            axes[0].set_title('Зависимость длины факела от расхода')
            axes[0].legend()

            # График зависимости диаметра факела от расхода
            axes[1].plot(consumptions, diameters, 'g-', linewidth=2)
            axes[1].plot(params['consumption'], Df_current, 'bo', label='Текущее значение')
            axes[1].set_xlabel('Расход, кг/с')
            axes[1].set_ylabel('Диаметр факела, м')
            axes[1].set_title('Зависимость диаметра факела от расхода')
            axes[1].legend()

            # Добавляем исходные данные
            info_text = (
                f"Исходные параметры:\n"
                f"Тип вещества: {self.substance_combo.currentText()}\n"
                f"Расход: {params['consumption']:.3f} кг/с"
            )
            # Добавляем информацию о параметрах на верхний график
            axes[0].text(0.02, 0.98, info_text,
                         transform=axes[0].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='white',
                                   alpha=0.8),
                         fontsize=8)

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
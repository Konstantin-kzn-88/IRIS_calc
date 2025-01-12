# widgets/evaporation_calculator.py
from PySide6.QtWidgets import (QDoubleSpinBox, QSpinBox, QComboBox,
                               QLabel, QHBoxLayout)
from widgets.base_calculator import BaseCalculatorWidget
from models.Liguid_Evaporation.liquid_evaporation import EvaporationModel
import numpy as np


class EvaporationCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета испарения жидкости"""

    # Предустановки для разных жидкостей
    PRESETS = {
        'water': {
            'name': 'Вода',
            'M': 0.018,
            'L': 2.26e6,
            'T_boil': 100,
        },
        'gasoline': {
            'name': 'Бензин',
            'M': 0.114,
            'L': 3.64e5,
            'T_boil': 125,
        },
        'oil': {
            'name': 'Нефть',
            'M': 0.170,
            'L': 2.30e5,
            'T_boil': 350,
        },
        'methanol': {
            'name': 'Метанол',
            'M': 0.032,
            'L': 1.10e6,
            'T_boil': 64.7,
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

        # Базовые параметры
        self.inputs['M'] = QDoubleSpinBox()
        self.inputs['M'].setRange(0.001, 1.0)
        self.inputs['M'].setValue(0.018)
        self.inputs['M'].setDecimals(3)
        self.input_layout.addRow("Молекулярная масса [кг/моль]:", self.inputs['M'])

        self.inputs['L'] = QDoubleSpinBox()
        self.inputs['L'].setRange(1e4, 1e7)
        self.inputs['L'].setValue(2.26e6)
        self.inputs['L'].setDecimals(0)
        self.input_layout.addRow("Удельная теплота испарения [Дж/кг]:", self.inputs['L'])

        # Температурные параметры
        for temp_param, label in [
            ('T_boil', 'Температура кипения [°C]'),
            ('T_init', 'Начальная температура [°C]'),
            ('T_amb', 'Температура окружающей среды [°C]'),
            ('T_ground', 'Температура поверхности [°C]')
        ]:
            self.inputs[temp_param] = QDoubleSpinBox()
            self.inputs[temp_param].setRange(-50, 500)
            self.inputs[temp_param].setDecimals(1)
            self.input_layout.addRow(label, self.inputs[temp_param])

        # Установка значений по умолчанию для температур
        self.inputs['T_boil'].setValue(100)
        self.inputs['T_init'].setValue(20)
        self.inputs['T_amb'].setValue(25)
        self.inputs['T_ground'].setValue(20)

        # Параметры среды
        self.inputs['wind_speed'] = QDoubleSpinBox()
        self.inputs['wind_speed'].setRange(0, 30)
        self.inputs['wind_speed'].setValue(3.0)
        self.inputs['wind_speed'].setDecimals(1)
        self.input_layout.addRow("Скорость ветра [м/с]:", self.inputs['wind_speed'])

        self.inputs['solar_flux'] = QDoubleSpinBox()
        self.inputs['solar_flux'].setRange(0, 2000)
        self.inputs['solar_flux'].setValue(500)
        self.inputs['solar_flux'].setDecimals(0)
        self.input_layout.addRow("Поток солнечного излучения [Вт/м²]:", self.inputs['solar_flux'])

        # Параметры пролива
        self.inputs['initial_mass'] = QDoubleSpinBox()
        self.inputs['initial_mass'].setRange(0.1, 10000)
        self.inputs['initial_mass'].setValue(100)
        self.inputs['initial_mass'].setDecimals(1)
        self.input_layout.addRow("Начальная масса [кг]:", self.inputs['initial_mass'])

        self.inputs['spill_area'] = QDoubleSpinBox()
        self.inputs['spill_area'].setRange(0.1, 1000)
        self.inputs['spill_area'].setValue(10)
        self.inputs['spill_area'].setDecimals(1)
        self.input_layout.addRow("Площадь пролива [м²]:", self.inputs['spill_area'])

        # Параметры моделирования
        self.simulation_time = QSpinBox()
        self.simulation_time.setRange(1, 24)
        self.simulation_time.setValue(1)
        self.input_layout.addRow("Время моделирования [ч]:", self.simulation_time)

        self.time_step = QDoubleSpinBox()
        self.time_step.setRange(1, 60)
        self.time_step.setValue(10)
        self.input_layout.addRow("Шаг по времени [с]:", self.time_step)

    def apply_preset(self, index):
        """Применение предустановки"""
        if index <= 0:
            return

        preset_id = self.preset_combo.currentData()
        preset = self.PRESETS[preset_id]

        for param, value in preset.items():
            if param != 'name' and param in self.inputs:
                self.inputs[param].setValue(value)

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def plot_results(self, results):
        """Построение графиков на виджете"""
        # Очищаем фигуру
        self.figure.clear()

        # Создаем подграфики
        (ax1, ax2, ax3) = self.figure.subplots(3, 1)

        t_minutes = results['t'] / 60

        # График температуры (конвертация в градусы Цельсия для отображения)
        ax1.plot(t_minutes, results['T'] - 273.15, 'b-', linewidth=2)
        ax1.set_xlabel('Время [мин]')
        ax1.set_ylabel('Температура [°C]')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Изменение температуры жидкости')

        # График массы
        ax2.plot(t_minutes, results['m'], 'g-', linewidth=2)
        ax2.set_xlabel('Время [мин]')
        ax2.set_ylabel('Масса [кг]')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Изменение массы жидкости')

        # График скорости испарения
        ax3.plot(t_minutes, np.abs(results['evap_rate']), 'r-', linewidth=2)
        ax3.set_xlabel('Время [мин]')
        ax3.set_ylabel('Скорость испарения [кг/с]')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.set_title('Скорость испарения')

        # Обновляем отображение
        self.figure.tight_layout()
        self.canvas.draw()

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров и создание модели
            params = self.get_params()
            model = EvaporationModel(params)

            # Получение параметров моделирования
            t_span = self.simulation_time.value() * 3600
            dt = self.time_step.value()

            # Выполнение моделирования
            results = model.simulate(t_span=t_span, dt=dt)

            # Вывод результатов
            self.output_text.append("Результаты моделирования:")
            self.output_text.append(
                f"Общая масса испарившейся жидкости: {results['total_evaporated']:.2f} кг")
            self.output_text.append(
                f"Средняя скорость испарения: {results['average_rate']:.4f} кг/с")
            self.output_text.append(
                f"Максимальная скорость испарения: {results['max_rate']:.4f} кг/с")
            self.output_text.append(
                f"Максимальная температура: {results['max_temp'] - 273.15:.1f} °C")

            # Построение графиков
            self.plot_results(results)

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")

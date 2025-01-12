# widgets/lpg_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
from widgets.base_calculator import BaseCalculatorWidget
from models.LPG_Evaporation.lpg_evaporation import (
    LPGSpillEvaporation,
    LPGProperties,
    SurfaceProperties
)


class LPGCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета испарения СУГ"""

    # Предустановки для разных типов СУГ
    LPG_PRESETS = {
        'propane': {
            'name': 'Пропан',
            'molecular_weight': 44.1,
            'boiling_temp': 231.1,  # -42°C
            'heat_capacity': 2500,
            'vaporization_heat': 428000,
            'liquid_density': 500
        },
        'butane': {
            'name': 'Бутан',
            'molecular_weight': 58.12,
            'boiling_temp': 272.7,  # -0.5°C
            'heat_capacity': 2400,
            'vaporization_heat': 385000,
            'liquid_density': 580
        },
        'propylene': {
            'name': 'Пропилен',
            'molecular_weight': 42.08,
            'boiling_temp': 225.5,  # -47.7°C
            'heat_capacity': 2530,
            'vaporization_heat': 437000,
            'liquid_density': 520
        }
    }

    # Предустановки для разных типов поверхностей
    SURFACE_PRESETS = {
        'concrete': {
            'name': 'Бетон',
            'thermal_conductivity': 1.28,
            'density': 2200,
            'heat_capacity': 840
        },
        'soil': {
            'name': 'Грунт',
            'thermal_conductivity': 0.8,
            'density': 1600,
            'heat_capacity': 1000
        },
        'asphalt': {
            'name': 'Асфальт',
            'thermal_conductivity': 0.7,
            'density': 2100,
            'heat_capacity': 920
        }
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор типа СУГ
        lpg_layout = QHBoxLayout()
        lpg_label = QLabel("Тип СУГ:")
        self.lpg_combo = QComboBox()
        self.lpg_combo.addItem("Выберите тип СУГ...")
        for preset_id, preset_data in self.LPG_PRESETS.items():
            self.lpg_combo.addItem(preset_data['name'], preset_id)
        self.lpg_combo.currentIndexChanged.connect(self.apply_lpg_preset)
        lpg_layout.addWidget(lpg_label)
        lpg_layout.addWidget(self.lpg_combo)
        lpg_layout.addStretch()
        self.input_layout.addRow(lpg_layout)

        # Выбор типа поверхности
        surface_layout = QHBoxLayout()
        surface_label = QLabel("Тип поверхности:")
        self.surface_combo = QComboBox()
        self.surface_combo.addItem("Выберите тип поверхности...")
        for preset_id, preset_data in self.SURFACE_PRESETS.items():
            self.surface_combo.addItem(preset_data['name'], preset_id)
        self.surface_combo.currentIndexChanged.connect(self.apply_surface_preset)
        surface_layout.addWidget(surface_label)
        surface_layout.addWidget(self.surface_combo)
        surface_layout.addStretch()
        self.input_layout.addRow(surface_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры СУГ
        for param, (label, value, decimals) in {
            'molecular_weight': ('Молекулярная масса [кг/кмоль]', 44.1, 3),
            'boiling_temp': ('Температура кипения [K]', 231.1, 1),
            'heat_capacity': ('Удельная теплоемкость [Дж/(кг·К)]', 2500, 0),
            'vaporization_heat': ('Теплота парообразования [Дж/кг]', 428000, 0),
            'liquid_density': ('Плотность жидкости [кг/м³]', 500, 0)
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0, 1e6)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры поверхности
        for param, (label, value, decimals) in {
            'thermal_conductivity': ('Теплопроводность [Вт/(м·К)]', 1.28, 2),
            'surface_density': ('Плотность поверхности [кг/м³]', 2200, 0),
            'surface_heat_capacity': ('Теплоемкость поверхности [Дж/(кг·К)]', 840, 0)
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0, 1e6)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры пролива
        for param, (label, value, decimals) in {
            'initial_mass': ('Начальная масса [кг]', 1000, 1),
            'spill_area': ('Площадь пролива [м²]', 100, 1),
            'initial_temp': ('Начальная температура [K]', 273.15, 2),
            'surface_temp': ('Температура поверхности [K]', 293.15, 2)
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0, 1e6)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_lpg_preset(self, index):
        """Применение предустановки СУГ"""
        if index <= 0:
            return

        preset_id = self.lpg_combo.currentData()
        preset = self.LPG_PRESETS[preset_id]

        # Обновляем значения в полях ввода
        self.inputs['molecular_weight'].setValue(preset['molecular_weight'])
        self.inputs['boiling_temp'].setValue(preset['boiling_temp'])
        self.inputs['heat_capacity'].setValue(preset['heat_capacity'])
        self.inputs['vaporization_heat'].setValue(preset['vaporization_heat'])
        self.inputs['liquid_density'].setValue(preset['liquid_density'])

    def apply_surface_preset(self, index):
        """Применение предустановки поверхности"""
        if index <= 0:
            return

        preset_id = self.surface_combo.currentData()
        preset = self.SURFACE_PRESETS[preset_id]

        # Обновляем значения в полях ввода
        self.inputs['thermal_conductivity'].setValue(preset['thermal_conductivity'])
        self.inputs['surface_density'].setValue(preset['density'])
        self.inputs['surface_heat_capacity'].setValue(preset['heat_capacity'])

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

            # Создание объектов свойств
            lpg = LPGProperties(
                name=self.lpg_combo.currentText(),
                molecular_weight=params['molecular_weight'],
                boiling_temp=params['boiling_temp'],
                heat_capacity=params['heat_capacity'],
                vaporization_heat=params['vaporization_heat'],
                liquid_density=params['liquid_density']
            )

            surface = SurfaceProperties(
                thermal_conductivity=params['thermal_conductivity'],
                density=params['surface_density'],
                heat_capacity=params['surface_heat_capacity']
            )

            # Создание модели и выполнение расчета
            model = LPGSpillEvaporation(
                lpg=lpg,
                surface=surface,
                initial_mass=params['initial_mass'],
                spill_area=params['spill_area'],
                initial_temp=params['initial_temp'],
                surface_temp=params['surface_temp']
            )

            # Расчет на 1 час
            times, masses, results = model.simulate(duration=3600)

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append(
                f"Начальная масса: {results['initial_mass']:.1f} кг")
            self.output_text.append(
                f"Доля мгновенного испарения: {results['flash_fraction'] * 100:.1f}%")
            self.output_text.append(
                f"Масса мгновенного испарения: {results['flash_mass']:.1f} кг")
            self.output_text.append(
                f"Время полного испарения: {results['total_time']:.1f} с")
            self.output_text.append(
                f"Средняя скорость испарения: {results['average_rate']:.3f} кг/с")

            # Построение графиков
            self.figure.clear()

            # Создаем подграфики (1 строка, 3 колонки)
            (ax1, ax2, ax3) = self.figure.subplots(1, 3)

            # График массы жидкости
            ax1.plot(times, masses, 'b-')
            ax1.set_xlabel('Время, с')
            ax1.set_ylabel('Масса жидкости, кг')
            ax1.grid(True)
            ax1.set_title('Изменение массы жидкости')

            # Добавляем текстовый блок с исходными данными
            info_text = (
                f"Исходные данные:\n"
                f"Вещество: {lpg.name}\n"
                f"Начальная масса: {params['initial_mass']:.1f} кг\n"
                f"Площадь пролива: {params['spill_area']:.1f} м²\n"
                f"Начальная температура: {params['initial_temp'] - 273.15:.1f}°C\n"
                f"Температура поверхности: {params['surface_temp'] - 273.15:.1f}°C\n"
                f"Мгновенное испарение: {results['flash_fraction'] * 100:.1f}%"
            )
            ax1.text(0.98, 0.98, info_text,
                     transform=ax1.transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=8)

            # График скорости испарения в логарифмическом масштабе
            evaporation_rates = results['evaporation_rates']
            ax2.plot(times[1:], evaporation_rates[1:], 'r-')
            ax2.set_yscale('log')
            ax2.set_xlabel('Время, с')
            ax2.set_ylabel('Скорость испарения, кг/с (лог)')
            ax2.grid(True, which="both", ls="-", alpha=0.2)
            ax2.grid(True, which="major", ls="-", alpha=0.5)
            ax2.set_title('Скорость испарения (логарифмическая шкала)')

            # График доли оставшейся массы
            ax3.plot(times, masses / masses[0] * 100, 'm-')
            ax3.set_xlabel('Время, с')
            ax3.set_ylabel('Оставшаяся масса, %')
            ax3.grid(True)
            ax3.set_title('Доля оставшейся массы')

            # Настраиваем layout
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")
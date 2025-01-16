# widgets/block_routes_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout, QCheckBox
from widgets.base_calculator import BaseCalculatorWidget
from models.Block_routes.block_routes import FireHazardCalculator, ToxicityParams, LiquidFuelParams
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


class BlockRoutesCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета времени блокирования путей эвакуации"""

    # Предустановки для разных жидкостей
    PRESETS = {
        'gasoline': {
            'name': 'Бензин',
            'heat_value': 44000,
            'smoke_value': 800,
            'burn_rate': 0.06,
            'density': 750,
            'co_yield': 0.1094,
            'co2_yield': 7.0,
            'hcl_yield': 0.0
        },
        'diesel': {
            'name': 'Дизельное топливо',
            'heat_value': 42700,
            'smoke_value': 620,
            'burn_rate': 0.04,
            'density': 850,
            'co_yield': 0.0885,
            'co2_yield': 3.2,
            'hcl_yield': 0.0
        },
        'kerosene': {
            'name': 'Керосин',
            'heat_value': 43500,
            'smoke_value': 700,
            'burn_rate': 0.05,
            'density': 800,
            'co_yield': 0.0950,
            'co2_yield': 5.0,
            'hcl_yield': 0.0
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

        # Параметры помещения
        for param, (label, value, decimals) in {
            'volume': ('Объем помещения [м³]', 3600.0, 1),
            'vent_area': ('Площадь вентиляции [м²]', 1.0, 2),
            'init_temp': ('Начальная температура [°C]', 20.0, 1),
            'burn_area': ('Начальная площадь горения [м²]', 1.0, 2),
            'height': ('Высота помещения [м]', 6.0, 1),
            'floor_area': ('Площадь пола [м²]', 600.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.1, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры материала
        for param, (label, value, decimals) in {
            'heat_value': ('Теплота сгорания [кДж/кг]', 44000, 0),
            'smoke_value': ('Дымообразующая способность [Нп×м²/кг]', 800, 0),
            'burn_rate': ('Массовая скорость выгорания [кг/(м²×с)]', 0.06, 3),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.001, 100000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры разлива
        for param, (label, value, decimals) in {
            'density': ('Плотность [кг/м³]', 750, 0),
            'thickness': ('Толщина разлива [м]', 0.005, 3),
            'spread_rate': ('Скорость растекания [м²/с]', 0.0001, 4)
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.0001, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры токсичности
        for param, (label, value, decimals) in {
            'co_yield': ('Выход CO [кг/кг]', 0.1094, 4),
            'co2_yield': ('Выход CO2 [кг/кг]', 7.0, 2),
            'hcl_yield': ('Выход HCl [кг/кг]', 0.0, 4)
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0, 100)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_preset(self, index):
        """Применение предустановки"""
        if index <= 0:
            return

        preset_id = self.preset_combo.currentData()
        preset = self.PRESETS[preset_id]

        # Обновляем значения в полях ввода для материала
        for param in ['heat_value', 'smoke_value', 'burn_rate', 'density',
                      'co_yield', 'co2_yield', 'hcl_yield']:
            if param in preset and param in self.inputs:
                self.inputs[param].setValue(preset[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        params = {key: widget.value() for key, widget in self.inputs.items()}

        # Конвертация температуры из °C в K
        params['init_temp'] += 273.15

        return params

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение параметров
            params = self.get_params()

            # Формирование параметров для расчета
            room_params = {
                'volume': params['volume'],
                'vent_area': params['vent_area'],
                'init_temp': params['init_temp'],
                'burn_area': params['burn_area'],
                'height': params['height'],
                'floor_area': params['floor_area']
            }

            material_params = {
                'heat_value': params['heat_value'],
                'smoke_value': params['smoke_value'],
                'burn_rate': params['burn_rate']
            }

            toxic_params = ToxicityParams(
                co_yield=params['co_yield'],
                co2_yield=params['co2_yield'],
                hcl_yield=params['hcl_yield']
            )

            liquid_params = LiquidFuelParams(
                density=params['density'],
                thickness=params['thickness'],
                spread_rate=params['spread_rate']
            )

            # Создание модели и выполнение расчета
            calculator = FireHazardCalculator(
                room_params=room_params,
                material_params=material_params,
                toxic_params=toxic_params,
                liquid_params=liquid_params
            )

            # Расчет времени блокирования
            results = calculator.calculate_blocking_time()

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)

            # Время блокирования по каждому фактору
            for factor, time in results.items():
                if time is not None:
                    if factor == 'blocking_time':
                        self.output_text.append(
                            f"\nМинимальное время блокирования: {time:.1f} с")
                    else:
                        factor_name = {
                            'temperature_time': 'температуре',
                            'visibility_time': 'потере видимости',
                            'co_time': 'CO',
                            'co2_time': 'CO2',
                            'hcl_time': 'HCl'
                        }.get(factor, factor)
                        self.output_text.append(
                            f"Время блокирования по {factor_name}: {time:.1f} с")
                else:
                    self.output_text.append(
                        f"Критическое значение по {factor} не достигнуто")

            # Дополнительная информация на момент блокирования
            t_block = results['blocking_time']
            if t_block is not None:
                self.output_text.append("\nПараметры на момент блокирования:")
                self.output_text.append("-" * 50)

                final_area = calculator.calculate_burn_area(t_block)
                burned = calculator.burned_mass(t_block)
                temp = calculator.temperature(t_block) - 273.15

                self.output_text.append(f"Площадь пожара: {final_area:.1f} м²")
                self.output_text.append(f"Масса выгоревшего топлива: {burned:.1f} кг")
                self.output_text.append(f"Температура в помещении: {temp:.1f}°C")

                # Концентрации токсичных веществ
                tox = calculator.toxic_concentrations(t_block)
                self.output_text.append("\nКонцентрации токсичных веществ:")
                self.output_text.append(f"CO: {tox['CO']:.4f} кг/м³")
                self.output_text.append(f"CO2: {tox['CO2']:.4f} кг/м³")
                self.output_text.append(f"HCl: {tox['HCl']:.4f} кг/м³")

            # Построение графиков
            calculator.plot_dynamics(save_path='temp_dynamics.png')

            # Загрузка и отображение графиков
            self.figure.clear()
            img = plt.imread('temp_dynamics.png')
            ax = self.figure.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")